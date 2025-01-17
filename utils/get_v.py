# traffic features
dim_s1, dim_s2, dim_s3 = 64, 32, 16
hidden_size1, hidden_size2, hidden_size3 = 300, 500, 600
dim_rho, dim_u = 256, 200
dim_c = 400
dropout = 0.2
use_selu = False
lengths = rn.edgeDis
dict_u, num_u = rn.roadid, len(rn.roadid)
dict_s1, num_s1 = rn.wayType, len(rn.wayType)
dict_s2, num_s2 = rn.lanes, len(rn.lanes)
dict_s3, num_s3 = rn.oneway, len(rn.oneway)
probrho = ProbRho(num_u, dim_u, dict_u, lengths,
                  num_s1, dim_s1, dict_s1,
                  num_s2, dim_s2, dict_s2,
                  num_s3, dim_s3, dict_s3,
                  hidden_size1, dim_rho,
                  dropout, use_selu, device).to(device)
probtraffic = ProbTraffic(4500, hidden_size2, dim_c,
                          dropout, use_selu).to(device)

traffic_path = './traffic/Porto_data_traffics.pkl'
with open(traffic_path, 'rb') as f:
    data_S = pickle.load(f)
data_S_tensor = torch.tensor(data_S, dtype=torch.float32).to(device)
if data_S_tensor.dim() == 2:
    data_S_tensor.unsqueeze_(0).unsqueeze_(0)
elif data_S_tensor.dim() == 3:
    data_S_tensor.unsqueeze_(0)

rho = probrho(num_u, dim_u, dict_u, lengths,
              num_s1, dim_s1, dict_s1,
              num_s2, dim_s2, dict_s2,
              num_s3, dim_s3, dict_s3,
              hidden_size1, dim_rho,
              dropout, use_selu, device).to(device)
c, mu_c, logvar_c = probtraffic(data_S_tensor)