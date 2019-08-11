# from rpy2 import robjects
# from rpy2.robjects import Formula, Environment
# from rpy2.robjects.vectors import IntVector, FloatVector,StrVector
# from rpy2.robjects.lib import grid
# from rpy2.robjects.packages import importr
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.conversion import localconverter
# from rpy2.robjects.conversion import localconverter
# from rpy2.robjects.packages import importr
# from rpy2.robjects.packages import importr, data
# from rpy2.rinterface_lib.embedded import RRuntimeError
# import warnings
# import math, datetime
# import rpy2.robjects.lib.ggplot2 as ggplot2
# import rpy2.robjects as ro
# import pandas as pd
# import rpy2.robjects as ro
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

def roc_plot_g(data):
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_from_pd_df = ro.conversion.py2rpy(data)

    r_from_pd_df
    rprint = robjects.globalenv.find("print")
    stats = importr('stats')
    grdevices = importr('grDevices')
    base = importr('base')
    datasets = importr('datasets')

    grid.activate()
    grdevices.png(file="Rplots.png", width=512, height=512)   ######  we can change the url
    gp = ggplot2.ggplot(r_from_pd_df)

    pp = gp + \
        ggplot2.aes_string(x='x', y='y') + \
        ggplot2.geom_point()
    pp.plot()
    grdevices.dev_off()
    filename = "Rplots.png"
    lena = mpimg.imread(filename)
    plt.figure(figsize=(15,10))
    plt.imshow(lena)
    plt.axis('off')
    plt.show()


#import uuid
#from rpy2.robjects.packages import importr
#from IPython.core.display import Image

#grdevices = importr('grDevices')
def ggplot_notebook(gg, width = 800, height = 600):
  fn = '{uuid}.png'.format(uuid = uuid.uuid4())
  grdevices.png(fn, width = width, height = height)
  gg.plot()
  grdevices.dev_off()
  filename=fn
  lena = mpimg.imread(filename)
  plt.figure(figsize=(15,10))
  plt.imshow(lena)
  plt.axis('off')
  plt.show()
