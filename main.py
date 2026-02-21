import os
from baler_classification import Classifier

if __name__ == "__main__":
    model = Classifier()

    src_dir = "./tests/a"
    src_files = os.listdir(src_dir)
    class_number, confidence = model.classify(
        bottom_path=os.path.join(src_dir, src_files[0]),
        top_path=os.path.join(src_dir, src_files[1]),
    )
    
    print(f"class_number: {class_number}, confidence: {confidence}")