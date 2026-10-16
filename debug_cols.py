import sys
sys.path.insert(0, '.')
from src.simulate import generate_bins, generate_fill_history, get_current_state, _load_road_graph
from src.predict import predict_all_bins
from src.priority import score_all_bins

G = _load_road_graph()
b = generate_bins(G)
h = generate_fill_history(b)
s = get_current_state(b, h)
print("State columns:", s.columns.tolist())
p = predict_all_bins(s, h)
print("Pred columns:", p.columns.tolist())
sc = score_all_bins(p)
print("Scored columns:", sc.columns.tolist())
