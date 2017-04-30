import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plotSaveLossAndAccuracy(pdfName, history):
    # summarize history for accuracy
    pp = PdfPages(pdfName)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    pp.savefig()
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    pp.savefig()
    pp.close()