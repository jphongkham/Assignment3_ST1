"""
Stage 1 EDA for Macroinvertebrate Image Analysis System.

This file scans the dataset, creates a dataframe, and generates all EDA outputs.
"""

# Import Path for file and folder handling
from pathlib import Path

# Import OpenCV for reading image files
import cv2

# Import matplotlib for creating graphs
import matplotlib.pyplot as plt

# Import NumPy for numerical calculations
import numpy as np

# Import pandas for dataframe creation and analysis
import pandas as pd

# Import seaborn for clearer visualisations
import seaborn as sns


# Constants used for image quality checks and EDA settings
VERY_SMALL_IMAGE_THRESHOLD = 100
UNUSUAL_ASPECT_RATIO_LOW = 0.5
UNUSUAL_ASPECT_RATIO_HIGH = 2.0
PIXEL_ANALYSIS_SAMPLE_SIZE = 50
SAMPLE_GRID_MAX_IMAGES = 12


class DatasetIndexer:
    """Scans image folders and creates a table of image information."""

    def __init__(self, dataset_folder):
        # Store the dataset folder path
        self.dataset_folder = Path(dataset_folder)

        # Store the file types supported by the program
        self.supported_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    def build_dataframe(self):
        """Read image files and return a pandas DataFrame."""

        # Create a list to store image information
        records = []

        # Loop through every file inside the dataset folder
        for file_path in self.dataset_folder.rglob("*"):

            # Skip files that are not supported image types
            if file_path.suffix.lower() not in self.supported_extensions:
                continue

            # Read the image using OpenCV
            image = cv2.imread(str(file_path))

            # If the image cannot be read, store it as unreadable
            if image is None:
                records.append({
                    "file_path": str(file_path),
                    "label": file_path.parent.name,
                    "width": 0,
                    "height": 0,
                    "channels": 0,
                    "file_extension": file_path.suffix.lower(),
                    "readable": False,
                    "aspect_ratio": 0
                })
                continue

            # Get the image height and width
            height, width = image.shape[:2]

            # Get the number of colour channels
            channels = image.shape[2] if len(image.shape) == 3 else 1

            # Calculate the image aspect ratio
            aspect_ratio = width / height if height > 0 else 0

            # Store useful image metadata
            records.append({
                "file_path": str(file_path),
                "label": file_path.parent.name,
                "width": width,
                "height": height,
                "channels": channels,
                "file_extension": file_path.suffix.lower(),
                "readable": True,
                "aspect_ratio": aspect_ratio
            })

        # Convert the records list into a pandas dataframe
        return pd.DataFrame(records)


class EDAService:
    """Creates summary statistics, visualisations, and reports."""

    def __init__(self, dataframe, output_folder):
        # Store the dataframe for later analysis
        self.dataframe = dataframe

        # Store and create the output folder
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Keep only readable images for visualisation outputs
        self.readable_dataframe = self.dataframe[self.dataframe["readable"] == True]

    def generate_all_outputs(self):
        """Generate all required EDA outputs and return their paths."""

        # Stop the program if the dataframe is empty
        if self.dataframe.empty:
            raise ValueError("The dataset index is empty.")

        # Run every EDA output function and store the generated file paths
        output_paths = [
            self.generate_dataset_summary(),
            self.generate_class_distribution_chart(),
            self.generate_image_size_distribution_chart(),
            self.generate_width_height_scatter_plot(),
            self.generate_sample_image_grid(),
            self.generate_width_by_class_boxplot(),
            self.generate_height_by_class_boxplot(),
            self.generate_pixel_intensity_histogram(),
            self.generate_image_quality_issues(),
            self.generate_class_imbalance_report(),
            self.generate_stage2_recommendations(),
        ]

        return output_paths

    def get_summary_dictionary(self):
        """Return key summary values for the GUI."""

        # Use readable images when calculating average image dimensions
        readable = self._require_readable_images()

        # Return key values so the GUI can display them
        return {
            "total_images": len(self.dataframe),
            "total_classes": self.dataframe["label"].nunique(),
            "average_width": round(readable["width"].mean(), 2),
            "average_height": round(readable["height"].mean(), 2),
        }

    def generate_dataset_summary(self):
        """Save a high-level dataset summary CSV."""

        # Get readable images and class counts
        readable = self._require_readable_images()
        class_counts = self.dataframe["label"].value_counts().sort_index()

        # List the file types found in the dataset
        supported_types = ", ".join(sorted(self.dataframe["file_extension"].unique()))

        # Store summary information as rows for a CSV file
        summary_rows = [
            ("total_images", len(self.dataframe)),
            ("total_classes", self.dataframe["label"].nunique()),
            ("images_per_class", self._format_class_counts(class_counts)),
            ("mean_width", self._safe_round(readable["width"].mean())),
            ("mean_height", self._safe_round(readable["height"].mean())),
            ("min_width", self._safe_int(readable["width"].min())),
            ("max_width", self._safe_int(readable["width"].max())),
            ("min_height", self._safe_int(readable["height"].min())),
            ("max_height", self._safe_int(readable["height"].max())),
            ("number_of_unreadable_files", int((~self.dataframe["readable"]).sum())),
            ("supported_file_types_found", supported_types),
        ]

        # Save the summary as a CSV file
        summary = pd.DataFrame(summary_rows, columns=["metric", "value"])
        output_path = self.output_folder / "dataset_summary.csv"
        summary.to_csv(output_path, index=False)

        return output_path

    def generate_class_distribution_chart(self):
        """Save a bar chart showing the number of images in each class."""

        # Count how many images are in each class
        class_counts = self.dataframe["label"].value_counts().sort_values(
            ascending=False
        )

        # Create a bar chart showing class balance
        plt.figure(figsize=(10, 6))
        sns.barplot(x=class_counts.index, y=class_counts.values)
        plt.title("Class Balance: Images per Macroinvertebrate Class")
        plt.xlabel("Class Label")
        plt.ylabel("Number of Images")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()

        # Save the chart as an image file
        output_path = self.output_folder / "class_distribution.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def generate_image_size_distribution_chart(self):
        """Save width and height histograms in one figure."""

        # Use only readable images for charting
        readable = self._require_readable_images()

        # Create two histograms: one for width and one for height
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.histplot(readable["width"], bins=20, ax=axes[0])
        axes[0].set_title("Image Width Distribution")
        axes[0].set_xlabel("Width in Pixels")

        sns.histplot(readable["height"], bins=20, ax=axes[1])
        axes[1].set_title("Image Height Distribution")
        axes[1].set_xlabel("Height in Pixels")

        fig.tight_layout()

        # Save the chart as an image file
        output_path = self.output_folder / "image_size_distribution.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        return output_path

    def generate_width_height_scatter_plot(self):
        """Save a scatter plot of image width versus height."""

        # Use only readable images
        readable = self._require_readable_images()

        # Create a scatter plot comparing width and height
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=readable,
            x="width",
            y="height",
            hue="label",
            alpha=0.75
        )
        plt.title("Image Width Versus Height")
        plt.xlabel("Width in Pixels")
        plt.ylabel("Height in Pixels")
        plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Save the scatter plot
        output_path = self.output_folder / "width_height_scatter.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def generate_sample_image_grid(self):
        """Save a grid of representative readable sample images."""

        # Select representative readable images
        readable = self._require_readable_images()
        samples = self._select_representative_samples(readable)

        # Work out the number of rows and columns required
        columns = min(4, len(samples))
        rows = int(np.ceil(len(samples) / columns))

        # Create the image grid
        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.4 * rows))
        axes_array = np.array(axes).reshape(-1)

        # Hide all axes before adding images
        for axis in axes_array:
            axis.axis("off")

        # Add each selected image to the grid
        for axis, (_, row) in zip(axes_array, samples.iterrows()):
            image = cv2.imread(str(row["file_path"]), cv2.IMREAD_COLOR)

            if image is None:
                continue

            # Convert OpenCV BGR colour format to RGB for matplotlib
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            axis.imshow(rgb_image)
            axis.set_title(str(row["label"]), fontsize=10)
            axis.axis("off")

        fig.suptitle("Representative Sample Images by Class")
        fig.tight_layout()

        # Save the sample image grid
        output_path = self.output_folder / "sample_image_grid.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        return output_path

    def generate_width_by_class_boxplot(self):
        """Save a boxplot comparing image widths by class."""

        # Use readable images only
        readable = self._require_readable_images()

        # Create a boxplot showing width differences by class
        plt.figure(figsize=(11, 6))
        sns.boxplot(data=readable, x="label", y="width")
        plt.title("Image Width by Class")
        plt.xlabel("Class Label")
        plt.ylabel("Width in Pixels")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()

        # Save the chart
        output_path = self.output_folder / "width_by_class_boxplot.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def generate_height_by_class_boxplot(self):
        """Save a boxplot comparing image heights by class."""

        # Use readable images only
        readable = self._require_readable_images()

        # Create a boxplot showing height differences by class
        plt.figure(figsize=(11, 6))
        sns.boxplot(data=readable, x="label", y="height")
        plt.title("Image Height by Class")
        plt.xlabel("Class Label")
        plt.ylabel("Height in Pixels")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()

        # Save the chart
        output_path = self.output_folder / "height_by_class_boxplot.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def generate_pixel_intensity_histogram(self):
        """Save a grayscale pixel intensity histogram from sampled images."""

        # Use a sample of readable images so the analysis does not take too long
        readable = self._require_readable_images()
        sample = readable.head(PIXEL_ANALYSIS_SAMPLE_SIZE)

        # Store grayscale pixel values
        intensity_values = []

        # Read sampled images as grayscale and collect pixel values
        for _, row in sample.iterrows():
            grayscale = cv2.imread(str(row["file_path"]), cv2.IMREAD_GRAYSCALE)

            if grayscale is not None:
                intensity_values.extend(grayscale.flatten().tolist())

        # Stop if no pixel data was collected
        if not intensity_values:
            raise ValueError("No readable pixels were available for analysis.")

        # Create the pixel intensity histogram
        plt.figure(figsize=(9, 6))
        sns.histplot(intensity_values, bins=50)
        plt.title("Sampled Grayscale Pixel Intensity Distribution")
        plt.xlabel("Pixel Intensity: 0 Dark to 255 Bright")
        plt.ylabel("Frequency")
        plt.tight_layout()

        # Save the histogram
        output_path = self.output_folder / "pixel_intensity_histogram.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def generate_image_quality_issues(self):
        """Save image quality flags to CSV."""

        # Store detected quality issue rows
        records = []

        # Check each image for simple quality issues
        for _, row in self.dataframe.iterrows():
            issues = []

            # Mark unreadable or corrupted images
            if not bool(row["readable"]):
                issues.append("unreadable_or_corrupted")

            # Mark very small images
            if bool(row["readable"]) and (
                row["width"] < VERY_SMALL_IMAGE_THRESHOLD
                or row["height"] < VERY_SMALL_IMAGE_THRESHOLD
            ):
                issues.append("very_small_image")

            # Mark unusual aspect ratio images
            if bool(row["readable"]) and (
                row["aspect_ratio"] < UNUSUAL_ASPECT_RATIO_LOW
                or row["aspect_ratio"] > UNUSUAL_ASPECT_RATIO_HIGH
            ):
                issues.append("unusual_aspect_ratio")

            # Save only images that have issues
            if issues:
                records.append({
                    "file_path": row["file_path"],
                    "label": row["label"],
                    "width": row["width"],
                    "height": row["height"],
                    "aspect_ratio": row["aspect_ratio"],
                    "issues": "; ".join(issues),
                })

        # Save quality issues to CSV
        issues_dataframe = pd.DataFrame(records)

        output_path = self.output_folder / "image_quality_issues.csv"
        issues_dataframe.to_csv(output_path, index=False)

        return output_path

    def generate_class_imbalance_report(self):
        """Save a written class imbalance report."""

        # Count images in each class
        class_counts = self.dataframe["label"].value_counts().sort_values(
            ascending=False
        )

        # Identify largest and smallest classes
        largest_class = class_counts.idxmax()
        smallest_class = class_counts.idxmin()
        largest_count = int(class_counts.max())
        smallest_count = int(class_counts.min())

        # Calculate imbalance ratio
        ratio = largest_count / smallest_count if smallest_count > 0 else 0

        # Create markdown report content
        report = [
            "# Class Imbalance Report",
            "",
            f"- Largest class: {largest_class} ({largest_count} images)",
            f"- Smallest class: {smallest_class} ({smallest_count} images)",
            f"- Imbalance ratio: {ratio:.2f}:1",
            "",
            "## Interpretation",
            "",
            "The dataset contains class imbalance. This means some classes have "
            "many more images than others.",
            "",
            "## Stage 2 Implication",
            "",
            "For future classification work, a stratified train/test split and "
            "class-level evaluation metrics would be useful.",
        ]

        # Save the report as a markdown file
        output_path = self.output_folder / "class_imbalance_report.md"
        output_path.write_text("\n".join(report), encoding="utf-8")

        return output_path

    def generate_stage2_recommendations(self):
        """Save EDA-based recommendations for future Stage 2 planning."""

        # Use EDA results to create future modelling recommendations
        readable = self._require_readable_images()
        class_counts = self.dataframe["label"].value_counts()

        # Calculate image size variation
        width_range = int(readable["width"].max() - readable["width"].min())
        height_range = int(readable["height"].max() - readable["height"].min())

        # Calculate class imbalance values
        smallest_class_count = int(class_counts.min())
        largest_class_count = int(class_counts.max())
        imbalance_ratio = largest_class_count / smallest_class_count

        # Create markdown report content
        report = [
            "# Stage 2 Planning Recommendations",
            "",
            "These recommendations explain how the Stage 1 EDA could support "
            "future classification work.",
            "",
            "## Resizing",
            "",
            f"Images vary by {width_range} pixels in width and "
            f"{height_range} pixels in height. Future modelling should resize "
            "images to a consistent input size.",
            "",
            "## Normalisation",
            "",
            "Future modelling should normalise pixel values so images are "
            "represented consistently.",
            "",
            "## Class Imbalance",
            "",
            f"The largest class has {largest_class_count} images and the "
            f"smallest class has {smallest_class_count} images. This gives an "
            f"imbalance ratio of {imbalance_ratio:.2f}:1.",
            "",
            "## Recommended Future Stage 2 Workflow",
            "",
            "1. Clean or remove unreadable images.",
            "2. Resize all images to the same dimensions.",
            "3. Normalise pixel values.",
            "4. Use a stratified train/test split.",
            "5. Evaluate results by class, not only by overall accuracy.",
        ]

        # Save the report as a markdown file
        output_path = self.output_folder / "stage2_recommendations.md"
        output_path.write_text("\n".join(report), encoding="utf-8")

        return output_path

    def _require_readable_images(self):
        """Return readable images or raise an error."""

        # Make sure there are readable images before making charts
        if self.readable_dataframe.empty:
            raise ValueError("No readable images were found for EDA charts.")

        return self.readable_dataframe

    def _select_representative_samples(self, readable):
        """Select up to one image per class, then fill remaining slots."""

        # Try to select one image from each class first
        per_class = readable.groupby("label", group_keys=False).head(1)

        # If there are enough classes, only use the maximum number needed
        if len(per_class) >= SAMPLE_GRID_MAX_IMAGES:
            return per_class.head(SAMPLE_GRID_MAX_IMAGES)

        # Fill remaining slots with other images
        remaining_slots = SAMPLE_GRID_MAX_IMAGES - len(per_class)
        remaining = readable.drop(per_class.index).head(remaining_slots)

        return pd.concat([per_class, remaining])

    def _format_class_counts(self, class_counts):
        """Format class counts into a readable summary value."""

        # Convert class counts into a readable string for the summary CSV
        return "; ".join(
            f"{label}: {count}" for label, count in class_counts.items()
        )

    def _safe_round(self, value):
        """Round a numeric value while handling missing data."""

        # Return 0.0 if value is missing
        if pd.isna(value):
            return 0.0

        return round(float(value), 2)

    def _safe_int(self, value):
        """Convert a numeric value to int while handling missing data."""

        # Return 0 if value is missing
        if pd.isna(value):
            return 0

        return int(value)


def run_stage1_eda(dataset_folder, output_folder):
    """Run the full Stage 1 EDA process."""

    # Convert folder paths into Path objects
    dataset_folder = Path(dataset_folder)
    output_folder = Path(output_folder)

    # Check that the dataset folder exists
    if not dataset_folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_folder}")

    # Build the dataframe from dataset images
    indexer = DatasetIndexer(dataset_folder)
    dataframe = indexer.build_dataframe()

    # Stop if no valid images were found
    if dataframe.empty:
        raise ValueError("No valid images were found in the dataset folder.")

    # Create output folder if needed
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save the full image index as a CSV file
    csv_path = output_folder / "dataset_index.csv"
    dataframe.to_csv(csv_path, index=False)

    # Create EDA service and generate all outputs
    eda = EDAService(dataframe, output_folder)
    output_paths = eda.generate_all_outputs()

    # Get summary values for the GUI
    summary = eda.get_summary_dictionary()

    # Return important output information to the GUI
    return {
        "summary": summary,
        "csv_path": csv_path,
        "output_paths": output_paths,
        "output_folder": output_folder,
    }


def main():
    """Run Stage 1 directly from Terminal."""

    # Default dataset location
    dataset_folder = Path.home() / "Downloads" / "stream_macroinvertebrates"

    # Default output location
    output_folder = Path.home() / "Downloads" / "eda_outputs"

    # Run the EDA workflow
    results = run_stage1_eda(dataset_folder, output_folder)

    # Print results in the terminal
    print("Stage 1 EDA complete.")
    print(f"Outputs saved to: {results['output_folder']}")
    print(results["summary"])


# Only run main when this file is executed directly
if __name__ == "__main__":
    main()
