from .preprocessing import Preprocessing, calculate_weights, transform_labels
from .model import SGDBinarizedModel, VanillaSGDBinarizedModel
from .model_selection import RegularizationGridSearch
from sklearn.pipeline import Pipeline


class Psupertime:

    def __init__(self, *args, **kwargs):
        self.preprocessing_ = Preprocessing
        self.model_ = VanillaSGDBinarizedModel
        self.cv_ = RegularizationGridSearch
        
        # data
        self.adata_ = None
        self.labels_ = None
        self.transformed_labels_ = None
        self.label_weights_ = None

    def fit(self, adata, labels):
        
        self.adata_ = Preprocessing().fit_transform(adata=adata)
        self.labels_ = labels
        self.tansformed_labels = transform_labels(labels)
        self.label_weights_ = calculate_weights(labels)
        
