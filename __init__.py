from .evolution import Evolution
from .genome_handler import GenomeHandler
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
__all__ = ['Evolution', 'GenomeHandler']