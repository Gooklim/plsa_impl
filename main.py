
from pprocess import Pprocess as PP
from plsa import pLSA 
import numpy as np
import logging


if __name__ == '__main__':
  # Setup logging -------------------------
  logging.basicConfig(filename='plsa.log', level=logging.INFO)
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  logging.getLogger('').addHandler(console)

  # Some basic configuration ---------------
  fname = './data.txt'
  fsw = './stopwords.txt'
  eps = 20.0
  key_word_size = 10

  # Preprocess -----------------------------
  pp = PP(fname, fsw)
  t_d = pp.get_t_d()

  V,D = t_d.shape
  logging.info("V = %d  D = %d" %(V,D))

  # Train model and get result -------------
  pmodel = pLSA()
  for z in range(3,(D+1)/2):
    (l, p_d_z, p_w_z, p_z)  = pmodel.train(t_d, z, eps)
    logging.info('z = %d eps = %f' %(z, l))
    for itz in range(z):
      logging.info('Topic %d' %itz)
      data = [(p_w_z[i][itz], i)  for i in range(len(p_w_z[:,itz])) ]
      data.sort(key=lambda tup:tup[0], reverse=True)
      for i in range(key_word_size):
        logging.info('%s : %.6f ' %(pp.get_word(data[i][1]), data[i][0] ))


