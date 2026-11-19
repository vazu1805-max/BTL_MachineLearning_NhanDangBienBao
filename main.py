import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from src.data import load_gtsrb
from src.features import transform_features
from src.models import build_models
from src.evaluate import evaluate_model, save_comparison

def parse_args():
    p = argparse.ArgumentParser(description="Traffic Sign Recognition - GTSRB")
    p.add_argument("--download", action="store_true", help="Cho phép tải bộ dữ liệu nếu chưa có")
    p.add_argument("--max-train", type=int, default=0, help="Giới hạn mẫu train; 0 dùng toàn bộ")
    p.add_argument("--max-test", type=int, default=0, help="Giới hạn mẫu test; 0 dùng toàn bộ")
    p.add_argument("--features", nargs="+", choices=["pixel", "hog"], default=["pixel", "hog"])
    return p.parse_args()

def main():
    args = parse_args()
    print("Đang nạp bộ dữ liệu GTSRB...")
    X_train, y_train, X_test, y_test = load_gtsrb(
        download=args.download, max_train=args.max_train, max_test=args.max_test
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, số lớp train: {len(np.unique(y_train))}")
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    results = []

    for feature_name in args.features:
        print(f"\n=== Đặc trưng: {feature_name.upper()} ===")
        Xtr = transform_features(X_train, feature_name)
        Xte = transform_features(X_test, feature_name)
        for model_name, model in build_models().items():
            print(f"Huấn luyện {model_name}...")
            model.fit(Xtr, y_train)
            metrics = evaluate_model(
                model, Xte, y_test, f"{model_name}_{feature_name}", "outputs"
            )
            metrics["feature"] = feature_name
            results.append(metrics)
            artifact = {
                "model": model, "feature": feature_name,
                "image_size": (32, 32), "classes": sorted(np.unique(y_train).tolist())
            }
            joblib.dump(artifact, f"models/{model_name}_{feature_name}.joblib")
            print(f"Kết quả: accuracy={metrics['accuracy']:.4f}, F1-macro={metrics['f1_macro']:.4f}")

    df = save_comparison(results, "outputs")
    print("\n=== BẢNG KẾT QUẢ ===")
    print(df.to_string(index=False))
    best = df.iloc[0]
    print(f"\nMô hình tốt nhất theo Macro F1: {best['model']} ({best['f1_macro']:.4f})")
    print("Đã lưu mô hình trong models/ và báo cáo trong outputs/.")

if __name__ == "__main__":
    main()
