# dataset 
[https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765]
# steps

- GenerateModel
- ConvNet-Optimizing
    - Optimizing
        - Define possible tests with layers, denses and conv https://www.youtube.com/watch?v=lV09_8432VA
- View performance logs: 
    ```sh
    $ tensorboard --logdir='logs/'
    ```
- convNetGenModel

# results

| Epochs | ValidationSplit | Results
| --- | --- | --- |
1 | 10% | 61%
3 | 10% | 72%
10 | 10% | 91%

# Models

#### 128x2
loss: 0.2710 - accuracy: 0.8878 - val_loss: 0.4937 - val_accuracy: 0.8016
