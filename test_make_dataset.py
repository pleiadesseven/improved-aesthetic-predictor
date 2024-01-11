import torch
from make_dataset import load_open_clip_model

def test_load_open_clip_model():
    # Test case 1: Test with default model name
    model, preprocess = load_open_clip_model()
    assert isinstance(model, torch.nn.Module)
    assert callable(preprocess)

    # Test case 2: Test with custom model name
    model, preprocess = load_open_clip_model(model_name="Custom-Model")
    assert isinstance(model, torch.nn.Module)
    assert callable(preprocess)

    # Add more test cases as needed

if __name__ == "__main__":
    test_load_open_clip_model()
    import torch
    from make_dataset import load_open_clip_model

def test_load_open_clip_model():
    # Test case 1: Test with default model name
    model, preprocess, custom_model = load_open_clip_model()
    assert isinstance(model, torch.nn.Module)
    assert callable(preprocess)
    assert custom_model is None

    # Test case 2: Test with custom model name
    model, preprocess, custom_model = load_open_clip_model(model_name="Custom-Model")
    assert isinstance(model, torch.nn.Module)
    assert callable(preprocess)
    assert custom_model is None

    # Add more test cases as needed

if __name__ == "__main__":
    test_load_open_clip_model()