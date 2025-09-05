import os
import pathlib
import tensorflow as tf
import numpy as np
import librosa

# --- 상수 정의 ---
DATASET_PATH = 'data'
SAMPLE_RATE = 8000
TARGET_SAMPLES = SAMPLE_RATE # 목표 오디오 길이: 1초 (8000 샘플)

def stretch_and_pad(waveform_tensor):
    """
    TensorFlow 텐서를 입력받아 NumPy 배열로 변환 후,
    librosa를 사용해 오디오를 1초 길이로 신축하고 길이를 정확히 맞춥니다.
    """
    waveform_np = waveform_tensor.numpy()
    target_len = TARGET_SAMPLES
    original_len = len(waveform_np)

    if original_len == 0:
        return np.zeros(target_len, dtype=np.float32)

    # 신축 비율 계산
    rate = original_len / target_len if target_len > 0 else 0
    if rate < 0.1: # 신축 비율이 너무 낮으면 오류가 날 수 있어 보정
        return np.zeros(target_len, dtype=np.float32)

    # librosa로 시간 신축 적용
    stretched_waveform = librosa.effects.time_stretch(y=waveform_np, rate=rate)

    # 길이를 정확히 TARGET_SAMPLES로 맞춤 (패딩 또는 자르기)
    final_len = len(stretched_waveform)
    if final_len < target_len:
        padding = target_len - final_len
        stretched_waveform = np.pad(stretched_waveform, (0, padding), 'constant')
    else:
        stretched_waveform = stretched_waveform[:target_len]
        
    return stretched_waveform.astype(np.float32)

def get_label_from_path(file_path):
    """ 파일명(e.g., '0_lucas_30.wav')에서 첫 글자를 라벨로 추출합니다. """
    parts = tf.strings.split(file_path, os.path.sep)
    filename = parts[-1]
    label_str = tf.strings.substr(filename, 0, 1)
    label = tf.strings.to_number(label_str, out_type=tf.int32)
    return label

def get_mel_spectrogram(waveform):
    """ 1초 길이의 오디오 파형을 로그 멜 스펙트로그램으로 변환합니다. """
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    
    num_spectrogram_bins = spectrogram.shape[-1]
    # 8kHz 샘플레이트에 맞는 파라미터로 조정
    lower_edge_hz, upper_edge_hz, num_mel_bins = 80.0, 4000.0, 64
    
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, SAMPLE_RATE, lower_edge_hz, upper_edge_hz)
    
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))
    
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, -1)
    
    return log_mel_spectrogram

def prepare_datasets(batch_size):
    """ 전체 데이터셋 준비 과정을 총괄합니다. """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"'{DATASET_PATH}' 폴더에 데이터 파일이 없습니다.")

    filenames = tf.io.gfile.glob(str(pathlib.Path(DATASET_PATH)) + '/*.wav')
    if not filenames:
        raise FileNotFoundError(f"'{DATASET_PATH}' 폴더에서 .wav 파일을 찾을 수 없습니다.")
        
    filenames = tf.random.shuffle(filenames)

    # 데이터셋을 Train, Validation, Test 세트로 8:1:1 분할
    num_samples = tf.shape(filenames)[0]
    num_train = tf.cast(tf.cast(num_samples, tf.float32) * 0.8, tf.int32)
    num_val = tf.cast(tf.cast(num_samples, tf.float32) * 0.1, tf.int32)

    train_files = filenames[:num_train]
    val_files = filenames[num_train : num_train + num_val]
    test_files = filenames[num_train + num_val :]

    def preprocess_file(file_path):
        # 1. 파일 경로에서 오디오 로드 및 시간 신축
        audio_binary = tf.io.read_file(file_path)
        waveform, _ = tf.audio.decode_wav(audio_binary)
        waveform = tf.squeeze(waveform, axis=-1)
        [stretched_waveform,] = tf.py_function(stretch_and_pad, [waveform], [tf.float32])
        stretched_waveform.set_shape([TARGET_SAMPLES])
        
        # 2. 멜 스펙트로그램으로 변환
        spectrogram = get_mel_spectrogram(stretched_waveform)
        
        # 3. 파일 경로에서 라벨 추출
        label = get_label_from_path(file_path)
        return spectrogram, label

    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = tf.data.Dataset.from_tensor_slices(train_files).map(preprocess_file, num_parallel_calls=AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices(val_files).map(preprocess_file, num_parallel_calls=AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices(test_files).map(preprocess_file, num_parallel_calls=AUTOTUNE)

    # 데이터셋 성능 최적화
    train_ds = train_ds.batch(batch_size).cache().prefetch(AUTOTUNE)
    val_ds = val_ds.batch(batch_size).cache().prefetch(AUTOTUNE)
    test_ds = test_ds.batch(batch_size).cache().prefetch(AUTOTUNE)

    print(f"데이터셋 준비 완료: {tf.shape(train_files)[0]} train, {tf.shape(val_files)[0]} val, {tf.shape(test_files)[0]} test samples.")
    return train_ds, val_ds, test_ds