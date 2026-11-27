import argparse
class Argument: 
    def __init__(self):
        self.parser = argparse.ArgumentParser()   
        
        # for Metasurface 
        self.parser.add_argument("--k", type=int, default=20) # kernel size
        self.parser.add_argument("--device", type=str, default="cuda:6")
        self.parser.add_argument("--f", type=float, default=0.8e-3)
        self.parser.add_argument("--R_cam", type=int, default=600)
        self.parser.add_argument("--C_cam", type=int, default=600) 
        self.parser.add_argument("--wvl", type=float, default=850e-9)
        self.parser.add_argument("--target_distance", type=float, default=20e-3)
        self.parser.add_argument("--min_dist", type=float, default=15e-3)
        self.parser.add_argument("--max_dist", type=float, default=30e-3)
        self.parser.add_argument("--R", type=int, default=5000)
        self.parser.add_argument("--C", type=int, default=5000)
        self.parser.add_argument("--MS_D", type=int, default= 2500)
        self.parser.add_argument("--pitch", type=float, default=400e-9)
        self.parser.add_argument("--pitch_light", type=float, default=400e-9)
        self.parser.add_argument("--pitch_cam", type=float, default=2.2e-6)
        self.parser.add_argument("--sensor_dist", type=float, default=0.8e-3) 
        self.parser.add_argument("--na", type=float, default=0.78)
        
        # Adaptive sampling
        
        self.parser.add_argument("--r_sample_min", type=int, default=3)
        self.parser.add_argument("--r_sample_max", type=int, default=9)
        self.parser.add_argument("--noise_gaussian_stddev", type=float, default=1e-2) 
        self.parser.add_argument("--phase_err_stddev", type=float, default=0.1)
        self.parser.add_argument("--N_sample", type=int, default=6)



        # FDTD-aware training 
        self.parser.add_argument("--phase2struct", type=str, default="./data/fdtd/phase2struct.txt")
        self.parser.add_argument("--proxy_phase", type=str, default="./data/fdtd/FDTD_phase.pkl")
        self.parser.add_argument("--proxy_T", type=str, default="./data/fdtd/FDTD_T.pkl")
        self.parser.add_argument("--fdtd_path", type=str, default='./data/fdtd/')
        
        

        # for training 
        self.parser.add_argument("--log_path", type=str, default="./data") 
        self.parser.add_argument('--train_dataset_path', type=str, default='./data/train')
        self.parser.add_argument('--valid_dataset_path', type=str, default='./data/valid')
        self.parser.add_argument("--lr_meta", type=float, default=1e-2)
        self.parser.add_argument("--lr_net", type=float, default=1e-4) 
        self.parser.add_argument("--mode", type=str, default="far")
        self.parser.add_argument("--net", type=str, default='no') 
        self.parser.add_argument("--use_net", type=bool, default=False)
        self.parser.add_argument("--is_nano", type=bool, default=True)
        self.parser.add_argument("--MS_train", type=bool, default=True)
        self.parser.add_argument("--init_MS", type=str, default='fresnel') # fresnel, random
        
        # for evaluation 
        self.parser.add_argument('--eval_epoch', type=int, default=0)
        
        
    def parse(self, data=[]):
        return self.parser.parse_args(data) 