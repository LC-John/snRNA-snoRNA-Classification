# snRNA-snoRNA-Classification

What's in the dir 'Dataset'?

  original datasets -- *.txt
    snRNA/snoRNA/tRNA/rRNA sequences of human/mouse.
  
  final.pkl.gz -- Human snRNA/snoRNA dataset
  
  final_m.pkl.gz -- Mouse snRNA/snoRNA dataset
  
  final_tr.pkl.gz -- Human rRNA/tRNA dataset
    Used to pollute snRNA/snoRNA dataset
    
  final_m_tr.pkl.gz -- Mouse rRNA/tRNA dataset
    same as above
  
  dirty.pkl.gz -- Human snRNA/snoRNA/trRNA
    Polluted dataset.
    
What's in the dir 'Code'?

  saved models (maybe) -- xxx_model_xxx
    CNN model parameters are saved in these dir's.
    
  dataset.py -- data operation
    All data related works are done in this script.
    Try not to do anything with it, if you don't understand -- it may deystroy the datasets in Dataset.
    
  CNN.py CNN_pred.py -- CNN model
    Not directly used in the experiment, but all other CNN files are derived from them. (It's really bothering to write a class, so I just copy and paste and modify several parts in each CNN file)
    
  CNN_cv.py CNN_cv_pred.py -- CNN model with CV
    k-fold CV applied while training, and the k classifiers vote when making predictions.
    use final.pkl.gz or final_m.pkl.gz when running these two scripts.
    
  dirty_CNN_cv.py -- CNN model with dirty samples
    add a new class, 'none-of-above'.
    use dirty.pkl.gz when running this script.
  
  plot.py, plot_cv.py -- plot curves
    to plot the learning curves generated when training.
