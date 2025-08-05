import pandas as pd
from PIL import Image as im
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import os

class GeoSeqDataLoader(Dataset):
    """
    DataLoader for video-gps datasets.

    The expected CSV file with the dataset information should have columns:
    - 'SEQ_ID'
    - 'IMG_FILE' for the image filename,
    - 'LAT' for latitude, and
    - 'LON' for longitude.
    """

    def __init__(self, dataset_file, transform=None):
        # self.dataset_folder = dataset_folder
        self.transform = transform
        self.videos, self.files, self.lats, self.lons = self.load_dataset(dataset_file)

    def load_dataset(self, dataset_file):
        try:


            #dataset_info = pd.read_csv(dataset_file)

            chunk_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
            num_chunks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
            chunk_file = dataset_file.replace(".", f"_{chunk_id}.")
            dataset_info = pd.read_csv(chunk_file)
            #vid_frames = 7

            #num_videos = dataset_info.shape[0]//vid_frames
            #chunk_videos_count = (num_videos//num_chunks) + 1
            #chunk_size = chunk_videos_count*vid_frames
            #start, end = chunk_id*chunk_size, (chunk_id+1)*chunk_size
            #dataset_info = dataset_info.iloc[start:end, :]
        except Exception as e:
            raise IOError(f"Error reading {dataset_file}: {e}")

        #print("chunk_id: ", chunk_id)
        #print(dataset_info)
        #exit()

        # MSLS_val_metadata = pd.read_csv('/home/c3-0/parthpk/Mapilliary_data/mapilliary_val_seq.csv')
        seq_ids = list(set(dataset_info["SEQ_ID"].to_list()))
        file_dict = {}
        lat_dict = {}
        lon_dict = {}
        lengths = []
        for seq_id in seq_ids:
            vid_metadata = dataset_info[dataset_info["SEQ_ID"] == seq_id]
            frame_files = vid_metadata["IMG_FILE"].to_list()
            lats, lons = vid_metadata["LAT"].to_list(), vid_metadata["LON"].to_list()
            file_dict[seq_id] = frame_files
            lat_dict[seq_id] = lats
            lon_dict[seq_id] = lons
            lengths.append(len(frame_files))
        #longest_vid_id = lengths.index(max(lengths))
        print(
            f"Loaded dataset from {dataset_file}. Videos: {len(file_dict)} Longest Video: {max(lengths)}"
        )
        #seq_ids = [seq_ids[longest_vid_id]] + seq_ids[:longest_vid_id] + seq_ids[longest_vid_id+1:]

        
        return seq_ids, file_dict, lat_dict, lon_dict

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        # print(f'{idx=}')
        vid_id = self.videos[idx]
        # print(vid_id)
        gps = torch.stack(
            [torch.Tensor(self.lats[vid_id]), torch.Tensor(self.lons[vid_id])], dim=1
        )
        frames = [im.open(img_path).convert("RGB") for img_path in self.files[vid_id]]

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        return vid_id, frames, gps
