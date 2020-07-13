from keras_wrapper.dataset import Dataset, saveDataset
from data_engine.prepare_data import keep_n_captions
from keras_wrapper.utils import decode_predictions_beam_search
from keras_wrapper.extra.read_write import list2file
from keras_wrapper.extra.evaluation import select
from config import load_parameters
from nmt_keras.model_zoo import TranslationModel
from nmt_keras.training import train_model
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.dataset import loadDataset
from keras_wrapper.extra.callbacks import PrintPerformanceMetricOnEpochEndOrEachNUpdates
import pyfiglet
import os

def start_training(use_gpu):

    ds = Dataset('tutorial_dataset', 'tutorial', silence=False)
    ds.setOutput(PATH + "train_correct.txt",
                 'train',
                 type='text',
                 id='target_text',
                 tokenization='tokenize_basic',
                 build_vocabulary=True,
                 pad_on_batch=True,
                 sample_weights=True,
                 max_text_len=100,
                 max_words=55000,
                 min_occ=1)

    ds.setOutput(PATH + "validation_correct.txt",
                 'val',
                 type='text',
                 id='target_text',
                 pad_on_batch=True,
                 tokenization='tokenize_basic',
                 sample_weights=True,
                 max_text_len=100,
                 max_words=0)

    ds.setInput(PATH + "train_error.txt",
                'train',
                type='text',
                id='source_text',
                pad_on_batch=True,
                tokenization='tokenize_basic',
                build_vocabulary=True,
                fill='end',
                max_text_len=100,
                max_words=55000,
                min_occ=1)

    ds.setInput(PATH + "validation_error.txt",
                'val',
                type='text',
                id='source_text',
                pad_on_batch=True,
                tokenization='tokenize_basic',
                fill='end',
                max_text_len=100,
                min_occ=1)

    """...and for the 'state_below' data. Note that: 1) The offset flat is set to 1, which means that the text will be shifted to the right 1 position. 2) During sampling time, we won't have this input. Hence, we 'hack' the dataset model by inserting an artificial input, of type 'ghost' for the validation split."""

    ds.setInput(PATH + "train_correct.txt",
                'train',
                type='text',
                id='state_below',
                required=False,
                tokenization='tokenize_basic',
                pad_on_batch=True,
                build_vocabulary='target_text',
                offset=1,
                fill='end',
                max_text_len=100,
                max_words=55000)
    ds.setInput(None,
                'val',
                type='ghost',
                id='state_below',
                required=False)

    """We can also keep the literal source words (for replacing unknown words)."""

    for split, input_text_filename in zip(['train', 'val'], [PATH + "train_error.txt", PATH + "validation_error.txt"]):
        ds.setRawInput(input_text_filename,
                      split,
                      type='file-name',
                      id='raw_source_text',
                      overwrite_split=True)

    """We also need to match the references with the inputs. Since we only have one reference per input sample, we set `repeat=1`."""

    keep_n_captions(ds, repeat=1, n=1, set_names=['val'])

    """Finally, we can save our dataset instance for using in other experiments:"""

    saveDataset(ds, PATH + "dataset")

    """## 2. Creating and training a Neural Translation Model
    Now, we'll create and train a Neural Machine Translation (NMT) model. Since there is a significant number of hyperparameters, we'll use the default ones, specified in the `config.py` file. Note that almost every hardcoded parameter is automatically set from config if we run  `main.py `.

    We'll create an `'AttentionRNNEncoderDecoder'` (a LSTM encoder-decoder with attention mechanism). Refer to the [`model_zoo.py`](https://github.com/lvapeab/nmt-keras/blob/master/nmt_keras/model_zoo.py) file for other models (e.g. Transformer). 

    So first, let's import the model and the hyperparameters. We'll also load the dataset we stored in the previous section (not necessary as it is in memory, but as a demonstration):
    """

    params = load_parameters()
    dataset = loadDataset(PATH + "dataset/Dataset_tutorial_dataset.pkl")

    """Since the number of words in the dataset may be unknown beforehand, we must update the params information according to the dataset instance:"""

    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len['source_text']
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len['target_text']
    params['USE_CUDNN'] = use_gpu
    params['N_GPUS'] = 2
    params['MAX_EPOCH'] = 1000
    params['EARLY_STOP'] = True
    params['PATIENCE'] = 10
    params['SAVE_EACH_EVALUATION'] = True
    params['STORE_PATH'] = PATH + "model/"
    params['BATCH_SIZE'] = 128
    params['ATTENTION_MODE'] = "add"
    params['N_LAYERS_ENCODER'] = 2
    params['N_LAYERS_DECODER'] = 2
    params['SOURCE_TEXT_EMBEDDING_SIZE'] = 512
    params['TARGET_TEXT_EMBEDDING_SIZE'] = 512
    params['SKIP_VECTORS_HIDDEN_SIZE'] = 512
    params['ATTENTION_SIZE'] = 512
    params['ENCODER_HIDDEN_SIZE'] = 512
    params['DECODER_HIDDEN_SIZE'] = 512
    params['ENCODER_RNN_TYPE'] = "LSTM"
    params['DECODER_RNN_TYPE'] = "ConditionalLSTM"
    params['METRICS'] = ['coco']
    params['KERAS_METRICS'] = ['perplexity']
    params['APPLY_DETOKENIZATION'] = True
    params['LENGTH_PENALTY'] = True
    params['LENGTH_NORM_FACTOR'] = 1.0
    params['BEAM_SIZE'] = 1
    params['BEAM_SEARCH'] = True
    params['PLOT_EVALUATION'] = True
    params['MAX_PLOT_Y'] = 1. 
    params['MODE'] = 'training' 
    params['TENSORBOARD'] = True

    result = pyfiglet.figlet_format("START TRAINING FROM SCRATCH".format(mode), font = "digital") 
    print(result)
    train_model(params, load_dataset = os.getcwd() + "/dataset/Dataset_tutorial_dataset.pkl")    

def resume_training(latest_epoch, use_gpu):

    params = load_parameters()
    params['MODEL_TYPE'] = 'AttentionRNNEncoderDecoder' 
    params['USE_CUDNN'] = use_gpu
    params['N_GPUS'] = 2
    params['MAX_EPOCH'] = latest_epoch + 1000
    params['BATCH_SIZE'] = 128
    params['EARLY_STOP'] = True
    params['PATIENCE'] = 10
    params['SAVE_EACH_EVALUATION'] = True
    params['STORE_PATH'] = PATH + "model/"
    params['ATTENTION_MODE'] = "add"
    params['N_LAYERS_ENCODER'] = 2
    params['N_LAYERS_DECODER'] = 2
    params['SOURCE_TEXT_EMBEDDING_SIZE'] = 512
    params['TARGET_TEXT_EMBEDDING_SIZE'] = 512
    params['SKIP_VECTORS_HIDDEN_SIZE'] = 512
    params['ATTENTION_SIZE'] = 512
    params['ENCODER_HIDDEN_SIZE'] = 512
    params['DECODER_HIDDEN_SIZE'] = 512
    params['ENCODER_RNN_TYPE'] = "LSTM"
    params['DECODER_RNN_TYPE'] = "ConditionalLSTM"
    params['METRICS'] = ['coco']
    params['KERAS_METRICS'] = ['perplexity']
    params['APPLY_DETOKENIZATION'] = True
    params['LENGTH_PENALTY'] = True
    params['LENGTH_NORM_FACTOR'] = 1.0
    params['RELOAD'] = latest_epoch
    params['BEAM_SIZE'] = 1
    params['BEAM_SEARCH'] = True
    params['PLOT_EVALUATION'] = True
    params['MAX_PLOT_Y'] = 1. 
    params['MODE'] = 'training' 
    params['TENSORBOARD'] = True

    result = pyfiglet.figlet_format("RESUME TRAINING".format(mode), font = "digital") 
    print(result)
    train_model(params, load_dataset = os.getcwd() + "/dataset/Dataset_tutorial_dataset.pkl")

def user_input_prediction(best_epoch):

    params = load_parameters()
    dataset = loadDataset(PATH + "dataset/Dataset_tutorial_dataset.pkl")
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]

    # Load model
    nmt_model = loadModel(PATH + 'model/', best_epoch)

    params_prediction = {
        'language': 'en',
        'tokenize_f': eval('dataset.' + 'tokenize_basic'),
        'beam_size': 12,
        'length_penalty': True,
        'length_norm_factor': 1.0,
        'optimized_search': True,
        'model_inputs': params['INPUTS_IDS_MODEL'],
        'model_outputs': params['OUTPUTS_IDS_MODEL'],
        'dataset_inputs':  params['INPUTS_IDS_DATASET'],
        'dataset_outputs':  params['OUTPUTS_IDS_DATASET'],
        'n_parallel_loaders': 1,
        'maxlen': 50,
        'model_inputs': ['source_text', 'state_below'],
        'model_outputs': ['target_text'],
        'dataset_inputs': ['source_text', 'state_below'],
        'dataset_outputs': ['target_text'],
        'normalize': True,
        'pos_unk': True,
        'heuristic': 0,
        'state_below_maxlen': 1,
        'predict_on_sets': ['test'],
        'verbose': 0,

      }
    result = pyfiglet.figlet_format("TESTING WITH USER INPUT".format(mode), font = "digital") 
    print(result)  
    while True:
        print("Input a sentence:")
        user_input = input()

        with open('user_input.txt', 'w') as f:
            f.write(user_input)
        dataset.setInput('user_input.txt',
                'test',
                type='text',
                id='source_text',
                pad_on_batch=True,
                tokenization='tokenize_basic',
                fill='end',
                max_text_len=100,
                min_occ=1,
                overwrite_split=True)

        dataset.setInput(None,
                    'test',
                    type='ghost',
                    id='state_below',
                    required=False,
                    overwrite_split=True)

        dataset.setRawInput('user_input.txt',
                      'test',
                      type='file-name',
                      id='raw_source_text',
                      overwrite_split=True)

        
        vocab = dataset.vocabulary['target_text']['idx2words']
        predictions = nmt_model.predictBeamSearchNet(dataset, params_prediction)['test']
        predictions = decode_predictions_beam_search(predictions[0],  # The first element of predictions contain the word indices.
                                                 vocab,
                                                 verbose=params['VERBOSE'])
        
        print(predictions[0])

PATH = ""
mode = "training"
use_gpu = True
model_dir = "./model/"
best_epoch = 13
eval_epoch = 2

def main():

    try:  
        os.mkdir(PATH + "model/")  
    except OSError as error:  
        print("Model directory already created.")  
    
    latest_validated_epoch = 0
    '''try:
        with open(PATH + "model/" + "val.coco", "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = ",")
            for row in csv_reader:
                if isinstance(row[0], int):
                    latest_validated_epoch = row[0]

    result = pyfiglet.figlet_format("Latest validated epoch is {}".format(latest_validated_epoch), font = "digital") 
    print(result)'''
    i = 0
    latest_epoch = 0
    epoch_list = os.listdir(PATH + model_dir)
    while True:
        if "".join(["epoch_", str(i), ".h5"]) in epoch_list:
            latest_epoch = i
        if i > 500:
            break
        i+=1

    result = pyfiglet.figlet_format("MODE: {}".format(mode), font = "digital") 
    print(result) 
    result = pyfiglet.figlet_format("Latest epoch is {}".format(latest_epoch), font = "digital") 
    print(result)

    if mode == "training":
        if latest_epoch == 0:
            start_training(use_gpu)
        elif latest_epoch != 0:
            resume_training(latest_epoch, use_gpu)
    elif mode == "testing":   
        user_input_prediction(best_epoch)

if __name__ == "__main__":
    main()