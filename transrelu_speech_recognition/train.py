import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# --- 추가할 라이브러리 ---
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
# --- 여기까지 ---

from data_loader import prepare_datasets
from model import build_model

# --- 하이퍼파라미터 및 설정 (이전과 동일) ---
BATCH_SIZE = 64
EPOCHS = 50
NUM_CLASSES = 10
MODEL_SAVE_PATH = 'saved_models/digit_recognizer_model.keras'
RESULT_PLOT_PATH = 'results/training_history_digits.png'
CONFUSION_MATRIX_PLOT_PATH = 'results/confusion_matrix.png' # 경로 추가

def main():
    # --- 폴더 생성 (이전과 동일) ---
    os.makedirs('results', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    # --- 1, 2, 3, 4번 과정 (이전과 동일) ---
    print("\n--- 1. 데이터셋 준비 시작 ---")
    train_ds, val_ds, test_ds = prepare_datasets(batch_size=BATCH_SIZE)
    for spectrogram, _ in train_ds.take(1):
        input_shape = spectrogram.shape[1:]
    print(f"감지된 Input Shape: {input_shape}")

    print("\n--- 2. 모델 빌드 ---")
    model = build_model(input_shape, NUM_CLASSES)
    model.summary()

    print("\n--- 3. 모델 컴파일 ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    print("\n--- 4. 모델 학습 시작 ---")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stopping])

    # --- 5. 학습 과정 시각화 (이전과 동일) ---
    print("\n--- 5. 결과 시각화 및 저장 ---")
    plot_history(history)

    # --- 6. 모델 평가 및 저장 (수정된 부분) ---
    print("\n--- 6. 상세 성능 평가 및 모델 저장 ---")
    
    # 6.1. 테스트 데이터셋에서 실제 라벨과 예측 라벨 추출
    y_true = []
    y_pred_probs = []
    
    for specs, labels in test_ds:
        y_true.extend(labels.numpy())
        y_pred_probs.extend(model.predict(specs, verbose=0))

    y_true = np.array(y_true)
    y_pred = np.argmax(np.array(y_pred_probs), axis=1)

    # 6.2. 성능 지표 계산 및 출력
    # Model Loss, Model Accuracy
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\n--- 최종 성능 지표 ---")
    print(f"Model Loss    : {test_loss:.4f}")
    print(f"Model Accuracy: {test_acc:.4f}")

    # F1 Score
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"F1 Score      : {f1:.4f}")

    # 6.3. Confusion Matrix 계산 및 시각화
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, num_classes=NUM_CLASSES)
    
    # 6.4. 모델 저장
    model.save(MODEL_SAVE_PATH)
    print(f"\n학습된 모델이 '{MODEL_SAVE_PATH}'에 저장되었습니다.")

def plot_history(history):
    # (이 함수는 이전 코드와 완전히 동일합니다)
    history_dict = history.history
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='o')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', marker='o')
    plt.plot(epochs_range, val_loss, label='Validation Loss', marker='o')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.savefig(RESULT_PLOT_PATH)
    print(f"학습 결과 그래프가 '{RESULT_PLOT_PATH}'에 저장되었습니다.")
    plt.show()

# --- 새로 추가된 함수 ---
def plot_confusion_matrix(cm, num_classes):
    """Confusion Matrix를 시각화하고 파일로 저장합니다."""
    plt.figure(figsize=(10, 8))
    class_names = [str(i) for i in range(num_classes)]
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(CONFUSION_MATRIX_PLOT_PATH)
    print(f"Confusion Matrix 그래프가 '{CONFUSION_MATRIX_PLOT_PATH}'에 저장되었습니다.")
    plt.show()

if __name__ == '__main__':
    main()