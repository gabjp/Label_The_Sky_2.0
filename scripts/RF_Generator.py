from ..Trainer.data_manager import load_data
from ..Trainer.sky_classifier import SkyClassifier
import numpy as np

def main():
    print("Loading Data")
    ds_name = 'clf_90_5_5'
    data = load_data(ds_name, True)
    print("Loaded Data")

    print("Generating Unified RF")
    #Generate unified RF:
    unified_rf = SkyClassifier('RF', "Unified_RF", True)
    unified_rf.build_model(n_estimator=100, bootstrap=False)
    unified_rf.train(data["tabular_train"], data["class_train"], " ")
    unified_rf.eval(data["tabular_val"], data["class_val"], "clf_90_5_5_val", wise_flags = data['wiseflags_val'])
    unified_rf.eval(data["tabular_test"], data["class_test"], "clf_90_5_5_test", wise_flags=data['wiseflags_test'])
    print("Generated Unified RF")

    print("Generating Without Wise RF")
    #Generate Without Wise RF:
    nowise_rf = SkyClassifier('RF', "Without_Wise_RF", True)
    nowise_rf.build_model(n_estimator=100, bootstrap=False)
    nowise_rf.train(data["tabular_train"][:,0:16], data["class_train"], "This model only uses SPLUS features")
    nowise_rf.eval(data["tabular_val"][:,0:16], data["class_val"], "clf_90_5_5_val", wise_flags = data['wiseflags_val'])
    nowise_rf.eval(data["tabular_test"][:,0:16], data["class_test"], "clf_90_5_5_test", wise_flags=data['wiseflags_test'])
    print("Generated Without Wise RF")

    print("Generating Only Wise RF")
    #Generate Only Wise RF:
    onlywise_rf = SkyClassifier('RF', "Without_Wise_RF", True)
    onlywise_rf.build_model(n_estimator=100, bootstrap=False)
    nowise_rf.train(data["tabular_train"][data['wiseflags_train'],:], data["class_train"][data['wiseflags_train'],:], "This model only uses objects with wise magnites")
    nowise_rf.eval(data["tabular_val"][data['wiseflags_val']], data["class_val"][data['wiseflags_val']], "clf_90_5_5_val_withwise")
    nowise_rf.eval(data["tabular_test"][data['wiseflags_test']], data["class_test"][data['wiseflags_test']], "clf_90_5_5_test_withwise")
    print("Generated Only Wise RF")

if __name__ == "__main__":
    main()