
import numpy as np
from torchvision.datasets import GTSRB
from torchvision import transforms
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (32, 32)
RANDOM_STATE = 42


def load_gtsrb(root="data", download=True, max_train=0, max_test=0):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=1),
    ])

    train_ds = GTSRB(
        root=root,
        split="train",
        download=download,
        transform=transform
    )

    test_ds = GTSRB(
        root=root,
        split="test",
        download=download,
        transform=transform
    )

    def convert(ds, limit=0):
        # Đọc nhãn của toàn bộ ảnh
        labels = np.array(
            [int(ds[i][1]) for i in range(len(ds))],
            dtype=np.int64
        )

        # Chọn mẫu ngẫu nhiên, giữ tỷ lệ các lớp
        if limit and limit < len(ds):
            indices, _ = train_test_split(
                np.arange(len(ds)),
                train_size=limit,
                stratify=labels,
                random_state=RANDOM_STATE
            )
            indices = np.sort(indices)
        else:
            indices = np.arange(len(ds))

        X, y = [], []

        for i in indices:
            img, label = ds[int(i)]
            X.append(np.asarray(img, dtype=np.uint8).reshape(-1))
            y.append(int(label))

        return (
            np.asarray(X, dtype=np.uint8),
            np.asarray(y, dtype=np.int64)
        )

    X_train, y_train = convert(train_ds, max_train)
    X_test, y_test = convert(test_ds, max_test)

    return X_train, y_train, X_test, y_test



def preprocess_uploaded_image(image, size=IMAGE_SIZE):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.Grayscale(num_output_channels=1),
    ])

    image = image.convert("RGB")
    image = transform(image)

    return np.asarray(
        image,
        dtype=np.uint8
    ).reshape(1, -1)

    image = image.convert("RGB").resize(size)
    gray = image.convert("L")
    return np.asarray(gray, dtype=np.uint8).reshape(1, -1)
