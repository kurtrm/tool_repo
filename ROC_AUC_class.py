"""Module containing a class that provides a convenient interface
for computing ROC/AUC and plotting the results.
"""
from scipy.integrate import trapz
from sklearn.metrics import auc

import matplotlib.pyplot as plt
import numpy as np


class ROC_AUC:
    """
    Provides an interace to compute ROC and AUC,
    and to plot their graphs.
    """
    def __init__(self, probabilities, labels):
        """
        Initialize ROC/AUC, takes an array of probabilities
        and an array of labels (actual values).
        """
        self.probabilities = probabilities
        self.labels = labels

    @property
    def roc(self) -> tuple:
        """
        Calculate the Receiver Operating Characteristics
        for the data and make it a property.
        """
        sorted_probs = np.sort(self.probabilities)
        actual_pos = len(self.labels[self.labels == 1])
        actual_neg = len(self.labels) - actual_pos
        fprs = np.empty(len(self.labels))
        tprs = np.empty(len(self.labels))
        for i, prob in enumerate(sorted_probs):
            partition = self.labels[self.probabilities > prob]
            true_positives = len(partition[partition == 1])
            false_positives = len(partition[partition == 0])
            true_pos_rate = true_positives / actual_pos
            false_pos_rate = false_positives / actual_neg
            tprs[i] = true_pos_rate
            fprs[i] = false_pos_rate

        return tprs, fprs

    def auc(self, method: str='scratch') -> float:
        """
        Calculate the Area Under the Curve for the ROC.
        For fun, the user can choose one of three ways to calculate
        it.

        Parameters
        ----------
        method -> str: User can choose between 'scratch', 'sklearn',
        and 'scipy'. 'scratch' returns a score from an implementation
        made from scratch. 'sklearn' returns a score from sklearn's
        roc_auc_score. 'scipy' returns the auc from the scipy.integrate.trapz
        function.

        Returns
        -------
        float auc score.
        """
        TPRs, FPRs = self.roc
        if method == 'scratch':
            return self._scratch_riemann(TPRs, FPRs)
        elif method == 'sklearn':
            return auc(FPRs, TPRs)
        else:
            return -trapz(TPRs, FPRs)  # Why negative? Dunno.

    def _scratch_riemann(self, tprs: np.array, fprs: np.array) -> float:
        """
        Private method for calculating a rough value for the
        auc.

        Parameters
        ----------
        tprs -> np.array: array of true positive rates.
        fprs -> np.array: array of false positive rates.

        Returns
        -------
        total -> float of auc score
        """
        total = 0
        p_tpr, p_fpr = 0, 0
        for tpr, fpr in zip(tprs, fprs):
            total += (tpr - p_tpr) * (fpr - p_fpr)
            p_tpr, p_fpr = tpr, fpr

        return total

    def plot(self):
        """
        Plot the ROC curve.
        """
        TPRs, FPRs = self.roc
        dashed_line = np.linspace(0, 1, 100)
        fig, ax = plt.subplots()
        ax.plot(FPRs, TPRs)
        ax.plot(dashed_line, dashed_line, '--')
        ax.set_title('ROC Curve')
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
