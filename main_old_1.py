#
import os
import pickle
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import tensorflow as tf
import subprocess
import os
import keras
import keras_nlp
import tensorflow as tf
from tensorboard.plugins import projector
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time
import json
import multiprocessing
from joblib import wrap_non_picklable_objects
class EnglishDataCleaning:
    ''' This class attempts to clean the raw english text by employing multiple cores. This class takes
        list of raw english strings as input and using clean_fast method it cleans the english text.
        clean_fast method also takes batch_size as parameter to perform cleaning in batches if data
        does not fit into the main memeory.
        
        Note: This method removes stop words.
        
        '''
    def __init__(self, data_dir, save_dir):
        self.data_dir = data_dir
        self.save_dir = save_dir

        
    def clean_fast(self,batch_size,start_batch_no=0):
        '''Method handling data in batches.'''
        with open('config.json', 'r') as f:
            config = json.load(f)
        if config["Last_batch_cleaned"] <= len(os.listdir(self.data_dir)):
            start_batch_no = config["Last_batch_cleaned"]
            list_of_files = os.listdir(self.data_dir)[start_batch_no:start_batch_no+batch_size]

            for i in range(len(list_of_files)):
                    
                print(f"Loading batch {start_batch_no+1} into memory...")
                data = []
                for j in tqdm(list_of_files):
                    with open(self.data_dir + f"/{j}", 'rb') as f:
                        for k in json.load(f):
                            for m in k['text'].split('.'):
                                data.append(m)
                print(f'Loaded batch {start_batch_no+1}.')
                print(f'Cleaning batch {start_batch_no+1}')
                self.fast_cleaning(start_batch_no,data)
                print(f'Cleaning of batch {start_batch_no+1} done.')
                print(f'Saving batch {start_batch_no+1}...')
                if not os.path.isdir(self.save_dir):
                    os.mkdir(self.save_dir)
                with open(f'{self.save_dir}/cleaned_batch_{start_batch_no+1}.pkl', 'wb') as f:
                    pickle.dump(data, f)
                print(f'Batch no {start_batch_no+1} saved successfully!')
            config["Last_batch_cleaned"] = start_batch_no + batch_size
            with open('config.json', 'w') as f:
                json.dump(config, f)
                
            
        else:
            print("Error! start_batch_no should be >= 1")

    def fast_cleaning(self, batch_no, data):
        '''Method using multiple cores to clean the text.'''
        def init_worker(mps, fps, cut):
            memorizedPaths, filepaths, cutoff = mps, fps, cut
            DG = 1
        def clean(text):
            import contractions
            from nltk.stem import WordNetLemmatizer
            from nltk.corpus import stopwords
            import re
            import nltk
            stop_words = stopwords.words('english')
            final_str = []
            regular_ex = r'[^a-zA-Z0-9\s\.]'
            regular_ex_1 = r'[$\n]'
            text = text.lower()
            text = re.sub(regular_ex,'',text)
            text = re.sub(regular_ex_1,'',text)
            lemmatizer = WordNetLemmatizer()
            tokenization = nltk.word_tokenize(text)
            for w in tokenization:
                w = contractions.fix(w)
                if w not in stop_words:
                    final_str.append(lemmatizer.lemmatize(w))
            return ' '.join(final_str)
        try:
            result = Parallel(n_jobs=-1, prefer="processes", verbose=6)(
            delayed(clean)(i) for i in tqdm(data))
        except Exception as e:
            print(e)




class DataPreprocessing:
    
    def __init__(self, data_dir, save_dir, vocab_path, batch_size):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.vocab_path = vocab_path
        self.vocab = None
        self.batch_size = batch_size
        
    def fast_make_save_sequences(self):
        with open('config.json', 'r') as f:
            config = json.load(f)
        list_of_files = os.listdir(self.data_dir)
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        print("Making voabulary...")
        with open(f'{self.vocab_path}', 'r') as f:
            vocab_ = f.read()
        vocab = {}
        for i in tqdm(vocab_.split()):
            if i not in vocab.keys():
                vocab[i] = len(vocab)
        self.vocab = vocab
        del vocab
        print("Making sequences...")
        self.fast_sequencing(list_of_files[config['last_batch_preprocessed']:config['last_batch_preprocessed']+self.batch_size])
        with open('config.json', 'w') as f:
            config['last_batch_preprocessed'] = config['last_batch_preprocessed']+self.batch_size
            json.dump(config,f)
        
    def fast_sequencing(self,list_of_files):
        manager = multiprocessing.Manager()
        done_files = manager.list()
        @wrap_non_picklable_objects
        def make_sequence(file):
            import os
            import pickle
            sequences = []
            with open(self.data_dir + f'/{file}', 'rb') as f:
                data = pickle.load(f)
                for m,j in (enumerate(data)):
                    t = []
                    for k in j.split():
                        if k in self.vocab.keys():
                            t.append(self.vocab[k])
                    sequences.append(t)
            
            done_files.append((f'{file}',sequences))
            if len(done_files) == self.batch_size:
                print("Started saving...")
                for i in tqdm(done_files):
                    with open(self.save_dir + f'sequences_batch_{i[0]}', 'wb') as f:
                        pickle.dump(i[1], f)
                done_files[:] = []
                print("Data saved successfully!")

        try:
            result = Parallel(n_jobs=-1, prefer="processes", verbose=6)(
            delayed(make_sequence)(i) for i in tqdm(list_of_files))
        except Exception as e:
            print(e)



class CustomPreprocessor:
    
    def __init__(self, data_dir, save_dir, vocab_path, mask_rate, seq_len, max_mask_per_seq, smallest_len_seq, batch_size):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.vocab_path = vocab_path
        self.mask_rate = mask_rate
        self.seq_len = seq_len
        self.max_mask_per_seq = max_mask_per_seq
        self.smallest_len_seq = smallest_len_seq
        self.batch_size = batch_size
        
        with open(f'{self.vocab_path}', 'r') as f:
            vocab_ = f.read()
        vocab = {}
        for i in tqdm(vocab_.split()):
            if i not in vocab.keys():
                vocab[i] = len(vocab)
        self.mask_id = vocab['[MASK]']
        del vocab
        
    def fast_make_save_MLM_dataset(self):
        list_of_files = os.listdir(self.data_dir)
        with open('config.json', 'r') as f:
            config = json.load(f)
        list_of_files = list_of_files[config['last_batch_custom_preprocessed']:config['last_batch_custom_preprocessed']+self.batch_size]
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        self.make_MLM_dataset(list_of_files)
        with open('config.json', 'w') as f:
            config['last_batch_custom_preprocessed'] = config['last_batch_custom_preprocessed']+self.batch_size
            json.dump(config, f)
        
    def make_MLM_dataset(self,list_of_files):
        manager = multiprocessing.Manager()
        done_files = manager.list()
        
        def make_MLM(file):
            import random
            import pickle
            print(file)
            with open(self.data_dir + f'{file}', 'rb') as f:
                sequences = pickle.load(f)
            
            mask_positions = []
            target_values = []
            weights = []
            new_sequences = []
            for m,i in tqdm(enumerate(sequences)):
                if len(i) > self.smallest_len_seq:
                    t = []
                    t_1 = []
                    t_2 = []
                    for k,j in enumerate(i):
                        if k < self.seq_len:
                            if random.random() <= self.mask_rate:
                                if len(t) < self.max_mask_per_seq:
                                    t.append(k)
                                    t_1.append(j)
                                    t_2.append(1)
                                    i[k] = self.mask_id
                    weights.append(t_2 if len(t_2) == self.max_mask_per_seq else (t_2 + [0]*(self.max_mask_per_seq-len(t_2))))
                    mask_positions.append(t if len(t) == self.max_mask_per_seq else (t+[0]*(self.max_mask_per_seq-len(t_2))))
                    target_values.append(t_1 if len(t_1) == self.max_mask_per_seq else (t_1+[0]*(self.max_mask_per_seq-len(t_1))))
                    new_sequences.append(i[:self.seq_len]+[0]*(self.seq_len-len(i[:self.seq_len])))

            done_files.append((f'{file}',({'tokens':tf.convert_to_tensor(new_sequences),'mask_positions':tf.convert_to_tensor(mask_positions,dtype='int32')},
            tf.convert_to_tensor(target_values), tf.convert_to_tensor(weights))))
            print("file name is: ",file)
            if len(done_files) == self.batch_size:
                print("Sarted saving MLM data...")
                for i in tqdm(done_files):
                    with open(self.save_dir + f'mlm_batch_{i[0]}', 'wb') as f:
                        pickle.dump(i[1], f)
                done_files[:] = []
                print("MLM Data saved successfully!")
        try:
            print("Making dataset for MLM modeling...")
            result = Parallel(n_jobs=-1, prefer="processes", verbose=6)(
            delayed(make_MLM)(i) for i in tqdm(list_of_files))
            print("Done...")
        except Exception as e:
            print(e)




class CustomCallback(keras.callbacks.Callback):
    
    def __init__(self, log_dir, vocab, encoder_model):
        self.log_dir = log_dir
        self.vocab = vocab
        self.encoder_model = encoder_model
    
    def on_epoch_begin(self, epoch, logs=None):
        print("Logging data...")
        self.log_tensorboard_projector_data()
        print("Starting tensorboard. Please wait for 10 to 15 seconds!!")
        self.tensorboard_reload()
        time.sleep(5)
        print("Tensorboard started.")
        
    def tensorboard_reload(self):
        print("Reloading tensorboard...")
        os.system('taskkill /IM "tensorboard.exe" /F')
        print("Please wait for 10 to 15 seconds!")
        pid = subprocess.Popen(['tensorboard', f'''--logdir={self.log_dir}'''])
    
    def log_tensorboard_projector_data(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        with open(os.path.join(self.log_dir, 'metadata.tsv'), "w") as f:
            for subwords in self.vocab.keys():
                f.write("{}\n".format(subwords))

            for unknown in range(1, len(self.vocab) - len(self.vocab)):
                f.write("unknown #{}\n".format(unknown))
        
        weights = tf.Variable(self.encoder_model.layers[1].get_weights()[0])

        checkpoint = tf.train.Checkpoint(embedding=weights)
        checkpoint.save(os.path.join(self.log_dir, "embedding.ckpt"))


        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()

        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(self.log_dir, config)

class LanguageModel():

    def __init__(self, seq_len, vocab_path, embedding_dim, num_layers,
                 intermediate_dim, num_heads, dropout, norm_epsilon, learning_rate, max_mask_per_seq, log_dir,
                transfer_learning_batch, models_save_path, model_checkpoint):

        self.seq_len = seq_len
        self.vocab_path = vocab_path
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.norm_epsilon = norm_epsilon
        self.learning_rate = learning_rate
        self.max_mask_per_seq = max_mask_per_seq
        self.log_dir = log_dir
        self.transfer_learning_batch = transfer_learning_batch
        self.models_save_path = models_save_path
        self.model_checkpoint = model_checkpoint
        self.encoder = None
        self.bert = None


        with open(self.vocab_path, 'r') as f:
            vocab_ = f.read()
        vocab = {}
        for i in (vocab_.split()):
            if i not in vocab.keys():
                vocab[i] = len(vocab)
        self.vocab = vocab
        del vocab
        
    def make_bert(self):
        inputs = keras.Input(shape=(self.seq_len,), dtype=tf.int32)
        embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=len(self.vocab),
        sequence_length=self.seq_len,
        embedding_dim=self.embedding_dim,
        )

        outputs = embedding_layer(inputs)
        outputs = keras.layers.LayerNormalization(epsilon=self.norm_epsilon)(outputs)
        outputs = keras.layers.Dropout(rate=self.dropout)(outputs)
        for i in range(1):
            outputs = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=self.intermediate_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            layer_norm_epsilon=self.norm_epsilon,
        )(outputs)

        encoder_model = keras.Model(inputs, outputs)
        self.encoder_model = encoder_model

        encoder_model.summary()
        inputs = {
        "tokens": keras.Input(shape=(self.seq_len,), dtype=tf.int32),
        "mask_positions": keras.Input(shape=(self.max_mask_per_seq,), dtype=tf.int32),
        }
        encoded_tokens = encoder_model(inputs["tokens"])
        outputs = keras_nlp.layers.MLMHead(
        embedding_weights=embedding_layer.token_embedding.embeddings, activation="softmax",
        )(encoded_tokens, mask_positions=inputs["mask_positions"])
        bert = keras.Model(inputs, outputs)
        bert.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        weighted_metrics=["sparse_categorical_accuracy"],
        jit_compile=True,
        )
        bert.summary()
        self.bert = bert
    
    def fit(self, model, train_ds, epochs, batch_size, start_batch_no=1):
        
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                config = json.load(f)
                start_batch_no = config['last_trained_batch']
    
        list_of_files = os.listdir(train_ds)[start_batch_no:start_batch_no+self.transfer_learning_batch]
        with open('config.json', 'w') as f:
            config['last_trained_batch'] = start_batch_no+self.transfer_learning_batch
            json.dump(config,f)
            
 
        tensorboard = TensorBoard(log_dir=self.log_dir, histogram_freq=0,
                          write_graph=True, write_images=False)
        if config['last_trained_batch']<= len(os.listdir(train_ds)):
            print("Strted transfer learning...")
            for i in tqdm(list_of_files):
                if not os.path.exists(self.model_checkpoint + f'/{start_batch_no+i}'):
                    os.mkdir(self.model_checkpoint + f'/{start_batch_no+i}')
                
                check_point = ModelCheckpoint(self.model_checkpoint + f'/{start_batch_no+i}',monitor='sparse_categorical_accuracy',mode='max')
                
                print(f"Starting training of batch {start_batch_no+1}...")
                data = []
                
                with open(train_ds + f"/{i}", 'rb') as f:
                    data.append(pickle.load(f))
                        
                        
                training_data = ({"tokens":tf.convert_to_tensor(tf.concat([d[0]['tokens'] for d in data],0)),
                                 "mask_positions":tf.convert_to_tensor(tf.concat([d[0]['mask_positions'] for d in data],0))},
                                tf.convert_to_tensor(tf.concat([d[1] for d in data],0)),
                                tf.convert_to_tensor(tf.concat([d[2] for d in data],0)))
                
                
                    
                self.bert.fit(training_data[0],(training_data[1],training_data[2]), epochs=epochs, batch_size=batch_size, 
                              callbacks=[CustomCallback(self.log_dir,self.vocab,self.encoder_model),tensorboard,check_point])
                print("saving the model...")
                self.bert.save(self.models_save_path + f'model_transfer_step_{start_batch_no+1}_with_epochs_{epochs}')
                print(f"Training of batch {start_batch_no+1} done. now procedding to the next transfer learning step.")
        else:
            print("Error!")    



language_model_handler = LanguageModel(seq_len=20,
                                      vocab_path='D:\Transformers Implementation\Language Model\\bert_vocab_uncased.txt',
                                      embedding_dim=256,
                                      num_layers=1,
                                      num_heads=4,
                                    intermediate_dim=256,
                                     dropout=0.1,
                                      norm_epsilon=1e-5,
                                      learning_rate=5e-4,
                                      max_mask_per_seq=3,
                                      log_dir='D:/Transformers Implementation/Language Model/Clean Project/logs/language_model/',
                                      transfer_learning_batch=1,
                                      models_save_path='D:/Transformers Implementation/Language Model/Clean Project/models/',
                                      model_checkpoint='D:/Transformers Implementation/Language Model/Clean Project/model_checkpoints/')



## Making final class to handle everything


class Main():
    def __init__(self, data_dir, cleaned_data_dir, preprocessed_save_dir,
     vocab_path, processing_batch_size, custom_preprocessing_save_dir,
     mask_rate, seq_len, max_mask_per_seq, smallest_len_seq, 
     embedding_dim, num_layers, intermediate_dim, num_heads, dropout, norm_epsilon,
     learning_rate, log_dir, transfer_learning_batch, models_save_path, model_checkpoint_path,
            epochs, batch_size
     ):
        self.data_dir = data_dir
        self.cleaned_data_dir = cleaned_data_dir
        self.processing_batch_size = processing_batch_size
        self.preprocessed_data_dir = cleaned_data_dir
        self.preprocessed_save_dir = preprocessed_save_dir
        self.vocab_path = vocab_path
        self.custom_preprocessing_data_dir = preprocessed_save_dir
        self.custom_preprocessing_save_dir = custom_preprocessing_save_dir
        self.mask_rate = mask_rate
        self.seq_len = seq_len
        self.max_mask_per_seq = max_mask_per_seq
        self.smallest_len_seq = smallest_len_seq
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.norm_epsilon = norm_epsilon
        self.learning_rate = learning_rate
        self.log_dir = log_dir
        self.transfer_learning_batch = transfer_learning_batch
        self.models_save_path = models_save_path
        self.model_checkpoint_path = model_checkpoint_path
        self.epochs = epochs
        self.batch_size = batch_size

    def cleaning(self):
        data_cleaner = EnglishDataCleaning(data_dir=self.data_dir,
                                    save_dir=self.cleaned_data_dir)
        data_cleaner.clean_fast(batch_size=self.processing_batch_size)
        
    def preprocessing(self):
        data_preprocessor = DataPreprocessing(data_dir=self.preprocessed_data_dir,
                                     save_dir=self.preprocessed_save_dir,
                                     vocab_path=self.vocab_path,batch_size=self.processing_batch_size)
        
        data_preprocessor.fast_make_save_sequences()
        
    def custom_preprocessing(self):
        custom_preprocessor = CustomPreprocessor(data_dir=self.custom_preprocessing_data_dir,
                           save_dir=self.custom_preprocessing_save_dir,
                            vocab_path=self.vocab_path,
                           mask_rate=self.mask_rate,
                           seq_len=self.seq_len,
                           max_mask_per_seq=self.max_mask_per_seq,smallest_len_seq=self.smallest_len_seq,
                           batch_size = self.processing_batch_size)
        
        custom_preprocessor.fast_make_save_MLM_dataset()

    def training(self):
        print("naman...")
        prev = 0
        while True:
            
            if os.listdir(self.custom_preprocessing_save_dir) != prev:
                prev = os.listdir(self.custom_preprocessing_save_dir)
                language_model_handler = LanguageModel(seq_len=self.seq_len,
                                              vocab_path=self.vocab_path,
                                              embedding_dim=self.embedding_dim,
                                              num_layers=self.num_layers,
                                              num_heads=self.num_heads,
                                            intermediate_dim=self.intermediate_dim,
                                             dropout=self.dropout,
                                              norm_epsilon=self.norm_epsilon,
                                              learning_rate=self.learning_rate,
                                              max_mask_per_seq=self.max_mask_per_seq,
                                              log_dir=self.log_dir,
                                              transfer_learning_batch=1,
                                              models_save_path=self.models_save_path,
                                              model_checkpoint=self.model_checkpoint_path)
                bert = language_model_handler.make_bert()
                language_model_handler.fit(bert,train_ds='D:/Transformers Implementation/Language Model/Clean Project/mlm/',
                                  epochs=self.epochs,batch_size=self.batch_size)
                
            else:
                pass
    
    def preprocessing_handler(self):
        
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                try:
                    config = json.load(f)
                except:
                    print("Try deleting config.json file!")
        else:
            config= dict()
            config["Last_batch_cleaned"] = 0
            config["last_batch_preprocessed"] = 0
            config["last_batch_custom_preprocessed"] = 0
            config["last_trained_batch"] = 0
            print(config)
            with open('config.json', 'w') as f:
                json.dump(config, f)

        list_of_files = os.listdir(self.data_dir)
        total_iterations = len(list_of_files)//self.processing_batch_size
        for i in range(total_iterations+1):
            self.cleaning()
            self.preprocessing()
            self.custom_preprocessing()

    def main(self):

        if not os.path.exists(self.cleaned_data_dir):
            os.mkdir(self.cleaned_data_dir)

        if not os.path.exists(self.preprocessed_save_dir):
            os.mkdir(self.preprocessed_save_dir)

        if not os.path.exists(self.custom_preprocessing_save_dir):
            os.mkdir(self.custom_preprocessing_save_dir)

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        if not os.path.exists(self.models_save_path):
            os.mkdir(self.models_save_path)

        if not os.path.exists(self.model_checkpoint_path):
            os.mkdir(self.model_checkpoint_path)
            
        p1 = multiprocessing.Process(target=self.preprocessing_handler)
        p2 = multiprocessing.Process(target=self.training)
    
        p1.start()
        p2.start()

    



Language_model = Main(data_dir='D:/Transformers Implementation/Language Model/Data/enwiki20201020',
           cleaned_data_dir='D:/Transformers Implementation/Language Model/Clean Project/cleaned_data/',
           processing_batch_size=1,
           preprocessed_save_dir='D:/Transformers Implementation/Language Model/Clean Project/sequences/',
           vocab_path = 'D:\Transformers Implementation\Language Model\\bert_vocab_uncased.txt',
           custom_preprocessing_save_dir = 'D:/Transformers Implementation/Language Model/Clean Project/mlm/',
           mask_rate = 0.25,
           seq_len = 20,
           max_mask_per_seq = 3,
           smallest_len_seq = 5,
           embedding_dim = 256,
           num_layers = 1,
           intermediate_dim = 512,
           num_heads = 4,
           dropout = 0.1,
           norm_epsilon = 1e-5,
           learning_rate = 5e-4,
           log_dir = 'D:/Transformers Implementation/Language Model/Clean Project/logs/',
           transfer_learning_batch = 5,
           models_save_path = 'D:/Transformers Implementation/Language Model/Clean Project/models/',
           model_checkpoint_path = 'D:/Transformers Implementation/Language Model/Clean Project/model_checkpoints/',
           epochs = 10,
           batch_size = 128
           )


if __name__ == '__main__':
    Language_model.main()





