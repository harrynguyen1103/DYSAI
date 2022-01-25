# DYSAI
Inception-v3 model to automatically identify histological features of oral epithelial dysplasia on whole-slide images (WSIs) from tongue H&amp;E stain specimens
Step by step 
#1. Scan histological slides and obtain WSIs by a scanner (Nanozoomer)
#2. Annotate WSIs by Nanozoomer Digital Pathology software and pathologists
#3. Convert annotated WSIs (at 10x) from ndpi format to jpg format
#4. Automatically crop WSis, then label, and classify image patches 
  guided_patchcollector0312.py
  reference_label.pkl
  referenced_labelmarker.py
#5. Prepare image-patch folders for training, validation, and testing dataset: patchCollection_utility.py
#6. Train models: inceptionv3_trainRevise.py
#7. Test models: inceptionv3_exam.py
#8. Compare models: compare_model0326.py
#9. Calculate mean absolute error: modelCompareApr01.py
