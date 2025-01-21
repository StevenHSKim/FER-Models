### POSTER Dataset configurations ###
POSTER_DATASET_CONFIGS = {
    'rafdb': {
        'num_classes': 7,
        'labels_name': ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sadness', 'Angry', "Neutral"],
        'data_path': '/userHome/userhome1/automl_undergraduate/FER_Models/Poster/POSTER/data/raf-basic/',
        'checkpoint_dir': '/userHome/userhome1/automl_undergraduate/FER_Models/Poster/POSTER/checkpoint',
    },
    'fer2013': {
        'num_classes': 7,
        'labels_name': ['Angry', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise', "Neutral"],
        'data_path': '/userHome/userhome1/automl_undergraduate/FER_Models/datasets/FER2013',
        'checkpoint_dir': '/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/POSTER/checkpoint',
    },
    'ferplus': {
        'num_classes': 8,
        'labels_name': ['Neutral', 'Happy', 'Surprise', 'Sadness', 'Angry', 'Disgust', 'Fear', 'Contempt'],
        'data_path': '/userHome/userhome1/automl_undergraduate/FER_Models/datasets/FERPlus',
        'checkpoint_dir': '/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/POSTER/checkpoint',
    },
    'expw': {
        'num_classes': 7,
        'labels_name': ['Angry', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise', 'Neutral'],
        'data_path': '/userHome/userhome1/automl_undergraduate/FER_Models/datasets/ExpW_dataset',
        'checkpoint_dir': '/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/POSTER/checkpoint',
    }
}

### DAN Dataset configurations ###
DAN_DATASET_CONFIGS = {
    'rafdb': {
        'num_classes': 7,
        'labels_name': ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral'],
        'data_path': '/userHome/userhome1/automl_undergraduate/FER_Models/DAN/datasets/raf-basic',
        'checkpoint_dir': '/userHome/userhome1/automl_undergraduate/FER_Models/DAN/checkpoints',
        'label_file': 'EmoLabel/list_patition_label.txt',
        'image_dir': 'Image/aligned'
    },
    'fer2013': {
        'num_classes': 7,
        'labels_name': ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral'],
        'data_path': '/userHome/userhome1/automl_undergraduate/FER_Models/datasets/FER2013',
        'checkpoint_dir': '/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/DAN/checkpoints',
        'label_file': 'fer2013_modified.csv'
    },
    'ferplus': {
        'num_classes': 8,
        'labels_name': ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt'],
        'data_path': '/userHome/userhome1/automl_undergraduate/FER_Models/datasets/FERPlus',
        'checkpoint_dir': '/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/DAN/checkpoints',
        'label_file': 'FERPlus_Label_modified.csv',
        'image_dir': 'FERPlus_Image'
    },
    'expw': {
        'num_classes': 7,
        'labels_name': ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral'],
        'data_path': '/userHome/userhome1/automl_undergraduate/FER_Models/datasets/ExpW_dataset',
        'checkpoint_dir': '/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/DAN/checkpoints',
        'label_file': 'label/label.lst',
        'image_dir': 'aligned_image'
    }
}


# DDAMFN Dataset configurations
DDAMFN_DATASET_CONFIGS = {
    'rafdb': {
        'num_classes': 7,
        'labels_name': ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral'],
        'data_path': '/userHome/userhome1/automl_undergraduate/FER_Models/DDAMFN/datasets/raf-basic',
        'checkpoint_dir': '/userHome/userhome1/automl_undergraduate/FER_Models/DDAMFN/checkpoints',
        'label_file': 'EmoLabel/list_patition_label.txt',
        'image_dir': 'Image/aligned',
        'subtract_label': True  # RAF-DB labels start from 1
    },
    'fer2013': {
        'num_classes': 7,
        'labels_name': ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral'],
        'data_path': '/userHome/userhome1/automl_undergraduate/FER_Models/datasets/FER2013',
        'checkpoint_dir': '/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/DDAMFN/checkpoints',
        'label_file': 'fer2013_modified.csv',
        'subtract_label': False
    },
    'ferplus': {
        'num_classes': 8,
        'labels_name': ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt'],
        'data_path': '/userHome/userhome1/automl_undergraduate/FER_Models/datasets/FERPlus',
        'checkpoint_dir': '/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/DDAMFN/checkpoints',
        'label_file': 'FERPlus_Label_modified.csv',
        'image_dir': 'FERPlus_Image',
        'subtract_label': True
    },
    'expw': {
        'num_classes': 7,
        'labels_name': ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral'],
        'data_path': '/userHome/userhome1/automl_undergraduate/FER_Models/datasets/ExpW_dataset',
        'checkpoint_dir': '/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/DDAMFN/checkpoints/expw',
        'label_file': 'label/label.lst',
        'image_dir': 'aligned_image',
        'subtract_label': False
    }
}


# AdaDF Dataset configurations
AdaDF_DATASET_CONFIGS = {
    'rafdb': {
        'num_classes': 7,
        'labels_name': ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral'],
        'data_path': '/userHome/userhome1/automl_undergraduate/FER_Models/datasets/raf-basic',
        'checkpoint_dir': '/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/Ada-DF/checkpoints/rafdb',
        'label_file': 'EmoLabel/list_patition_label.txt',
        'image_dir': 'Image/aligned',
        'subtract_label': True,
        'is_csv': True,
        'csv_sep': ' ',
        'name_col': 'name',
        'label_col': 'label'
    },
    'fer2013': {
        'num_classes': 7,
        'labels_name': ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral'],
        'data_path': '/userHome/userhome1/automl_undergraduate/FER_Models/datasets/FER2013/fer2013_modified.csv',
        'checkpoint_dir': '/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/Ada-DF/checkpoints/fer2013',
        'is_csv': True,
        'pixels_col': 'pixels',
        'label_col': 'emotion'
    },
    'ferplus': {
        'num_classes': 8,
        'labels_name': ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt'],
        'data_path': '/userHome/userhome1/automl_undergraduate/FER_Models/datasets/FERPlus/FERPlus_Label_modified.csv',
        'checkpoint_dir': '/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/Ada-DF/checkpoints/ferplus',
        'image_dir': '/userHome/userhome1/automl_undergraduate/FER_Models/datasets/FERPlus/FERPlus_Image',
        'is_csv': True,
        'name_col': 'Image name',
        'label_col': 'label'
    },
    'expw': {
        'num_classes': 7,
        'labels_name': ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral'],
        'data_path': '/userHome/userhome1/automl_undergraduate/FER_Models/datasets/ExpW_dataset/label/label.lst',
        'checkpoint_dir': '/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/Ada-DF/checkpoints/expw',
        'image_dir': '/userHome/userhome1/automl_undergraduate/FER_Models/datasets/ExpW_dataset/aligned_image',
        'is_csv': False
    }
}