from main import Main

if __name__ == '__main__':
    Language_model = Main(data_dir='D:/Transformers Implementation/Language Model/Data/enwiki20201020',
           cleaned_data_dir='D:/Transformers Implementation/Language Model/Clean Project/cleaned_data/',
           processing_batch_size=5,
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
           epochs = 5,
           batch_size = 32,
           projector_dir = 'D:/Transformers Implementation/Language Model/Clean Project/projector/',
           )
    Language_model.main()
