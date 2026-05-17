# Stage 2 Planning Recommendations

These recommendations explain how the Stage 1 EDA could support future classification work.

## Resizing

Images vary by 326 pixels in width and 326 pixels in height. Future modelling should resize images to a consistent input size.

## Normalisation

Future modelling should normalise pixel values so images are represented consistently.

## Class Imbalance

The largest class has 987 images and the smallest class has 9 images. This gives an imbalance ratio of 109.67:1.

## Recommended Future Stage 2 Workflow

1. Clean or remove unreadable images.
2. Resize all images to the same dimensions.
3. Normalise pixel values.
4. Use a stratified train/test split.
5. Evaluate results by class, not only by overall accuracy.