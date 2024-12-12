from model import load_all_npy_files, create_model, train_save_gui, load_and_evaluate_model_gui,evaluate_model_gui
from model import ChannelAttention, SpatialAttention, DisplacementPredictor

data_path = r"C:\Users\Lenovo\Desktop\CSE 583 Software Development for Data Scientists\Project\Pylint_improvement\dataset1_sample\TestBatch"
model_path = r"C:\Users\Lenovo\Desktop\CSE 583 Software Development for Data Scientists\Project\Pylint_improvement\saved_model\trained_displacement_predictor_full_model.pth"
pred_save_dir = r"C:\Users\Lenovo\Desktop\CSE 583 Software Development for Data Scientists\Project\Pylint_improvement\save_pred"

load_and_evaluate_model_gui(model_path, data_path, pred_save_dir)
