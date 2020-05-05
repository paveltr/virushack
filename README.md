VirusHack API
=============

This project contains API for predicting diagnoses based on text descriptions

![alt text](image.png)


Sample call
=============

```
curl  -d '{"symptomps" : "Потливость насморк", "age" : "25", "gender" : "мужской"}' -H "Content-Type: application/json" -X POST  http://35.221.3.152/predict
```
