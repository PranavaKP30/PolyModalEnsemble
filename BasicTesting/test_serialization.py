import sys
import os
import numpy as np
import pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MainModel import mainModel, dataIntegration

def test_serialization():
    N_train, N_test = 40, 10
    import random
    import torch
    np.random.seed(11)
    rng_state_before = np.random.get_state()
    py_rng_state_before = random.getstate()
    torch_rng_state_before = torch.get_rng_state()
    loader = dataIntegration.GenericMultiModalDataLoader()
    loader.add_modality_split('text', np.random.randn(N_train, 5), np.random.randn(N_test, 5))
    loader.add_modality_split('image', np.random.randn(N_train, 5), np.random.randn(N_test, 5))
    labels_train = np.random.randint(0, 2, N_train)
    labels_test = np.random.randint(0, 2, N_test)
    loader.add_labels_split(labels_train, labels_test)
    model = mainModel.MultiModalEnsembleModel(data_loader=loader, n_bags=2, epochs=2, batch_size=8)
    model.load_and_integrate_data()
    model.fit()
    pred1 = model.predict().predictions
    # Serialize
    with open('model_serialization_test.pkl', 'wb') as f:
        pickle.dump(model, f)
    # Restore all RNG states before deserialization
    np.random.set_state(rng_state_before)
    random.setstate(py_rng_state_before)
    torch.set_rng_state(torch_rng_state_before)
    # Deserialize
    with open('model_serialization_test.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    pred2 = loaded_model.predict().predictions
    identical = np.array_equal(pred1, pred2)
    print(f"Predictions identical after serialization: {identical}")
    print(f"Original predictions: {pred1}")
    print(f"Reloaded predictions: {pred2}")
    os.remove('model_serialization_test.pkl')

if __name__ == "__main__":
    print("Testing model serialization...")
    test_serialization()
