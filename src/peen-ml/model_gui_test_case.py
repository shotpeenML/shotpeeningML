from model import load_and_evaluate_model_gui
# from model import load_all_npy_files, create_model, train_save_gui
# from model import ChannelAttention, SpatialAttention, DisplacementPredictor

DATA_PATH = r"C:\Users\Lenovo\Desktop\CSE 583 Software Development for Data Scientists\Project\Pylint_improvement\dataset1_sample\TestBatch"
MODEL_PATH = r"C:\Users\Lenovo\Desktop\CSE 583 Software Development for Data Scientists\Project\Pylint_improvement\saved_model\trained_displacement_predictor_full_model.pth"
PRED_SAVE_DIR = r"C:\Users\Lenovo\Desktop\CSE 583 Software Development for Data Scientists\Project\Pylint_improvement\save_pred"

load_and_evaluate_model_gui(MODEL_PATH, DATA_PATH, PRED_SAVE_DIR)
