import json
import pdb

def get_anet_meta(json_file, subset='train_and_val'):
    data = json.load(open(json_file, 'r'))
    texonomy_data = data['taxonomy']
    database_data = data['database']


    if subset == 'train':
        subset_data = {
            k: v for k, v in database_data.items() if v['subset'] == 'training'
        }
    elif subset == 'val':
        subset_data = {
            k: v
            for k, v in database_data.items()
            if v['subset'] == 'validation'
        }
    elif subset == 'train_and_val':
        subset_data = {
            k: v
            for k, v in database_data.items()
            if v['subset'] in ['training', 'validation']
        }
    elif subset == 'test':
        subset_data = {
            k: v for k, v in database_data.items() if v['subset'] == 'testing'
        }

    dataset_dict = {}

    return subset_data

if __name__ == '__main__':
    get_anet_meta('/MD1400/jhseon/datasets/activity-net-v1.3/activity_net.v1-3.min.json')


