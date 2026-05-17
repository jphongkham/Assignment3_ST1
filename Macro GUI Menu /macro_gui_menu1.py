"""
Stage 3 GUI Menu for Macroinvertebrate Image Analysis System.

This file creates the Tkinter GUI and connects it to the Stage 1 EDA code.
"""

# Import Path for handling folder and file locations
from pathlib import Path

# Import Tkinter for creating the graphical user interface
import tkinter as tk

# Import file selection windows and pop-up message boxes
from tkinter import filedialog, messagebox

# Import Pillow for image display inside the GUI
from PIL import Image, ImageTk

# Import the Stage 1 EDA function from the analysis module
from stage1_eda import run_stage1_eda


class MacroGuiMenu:
    """Main GUI menu for the macroinvertebrate image system."""

    def __init__(self, root):

        # Store the main Tkinter window
        self.root = root

        # Set the GUI window title
        self.root.title("Macroinvertebrate Image Analysis System")

        # Set the window size
        self.root.geometry("950x750")

        # Store the currently selected image path
        self.selected_image_path = None

        # Store the preview image used in Tkinter
        self.preview_image = None

        # Default dataset folder location
        self.dataset_folder = Path.home() / "Downloads" / "stream_macroinvertebrates"

        # Folder used to save generated EDA outputs
        self.output_folder = Path.home() / "Downloads" / "eda_outputs"

        # Create all GUI widgets and buttons
        self.create_widgets()

    def create_widgets(self):
        """Create all buttons, labels, and menu items."""

        # Main title label
        tk.Label(
            self.root,
            text="Macroinvertebrate Image Analysis System",
            font=("Arial", 20, "bold")
        ).pack(pady=15)

        # Subtitle label
        tk.Label(
            self.root,
            text="Stage 1 EDA connected to Stage 3 Tkinter GUI",
            font=("Arial", 12)
        ).pack(pady=5)

        # Create a frame to organise the menu buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=15)

        # Button used to select the dataset folder
        tk.Button(
            button_frame,
            text="1. Select Dataset Folder",
            width=30,
            command=self.select_dataset_folder
        ).grid(row=0, column=0, padx=10, pady=7)

        # Button used to run Stage 1 EDA analysis
        tk.Button(
            button_frame,
            text="2. Generate Stage 1 EDA",
            width=30,
            command=self.generate_eda_outputs
        ).grid(row=1, column=0, padx=10, pady=7)

        # Button used to choose and preview an image
        tk.Button(
            button_frame,
            text="3. Choose Image",
            width=30,
            command=self.choose_image
        ).grid(row=2, column=0, padx=10, pady=7)

        # Button used to show image metadata
        tk.Button(
            button_frame,
            text="4. Show Image Details",
            width=30,
            command=self.show_image_details
        ).grid(row=3, column=0, padx=10, pady=7)

        # Button used to show the EDA output folder
        tk.Button(
            button_frame,
            text="5. Show EDA Output Location",
            width=30,
            command=self.show_output_folder
        ).grid(row=4, column=0, padx=10, pady=7)

        # Button used to close the application
        tk.Button(
            button_frame,
            text="6. Exit",
            width=30,
            command=self.root.quit
        ).grid(row=5, column=0, padx=10, pady=7)

        # Image preview area
        self.image_label = tk.Label(
            self.root,
            text="No image selected",
            relief="solid"
        )
        self.image_label.pack(pady=10)

        # Label used to display results and status messages
        self.result_label = tk.Label(
            self.root,
            text=(
                "Select the dataset folder, generate EDA outputs, "
                "or choose an image to preview."
            ),
            font=("Arial", 12),
            wraplength=850,
            justify="left"
        )
        self.result_label.pack(pady=10)

    def select_dataset_folder(self):
        """Allow the user to select the dataset folder."""

        # Open a folder selection window
        folder_path = filedialog.askdirectory(
            title="Select stream_macroinvertebrates dataset folder"
        )

        # Stop if the user cancels selection
        if not folder_path:
            return

        # Save the selected dataset folder path
        self.dataset_folder = Path(folder_path)

        # Display selected dataset location
        self.result_label.config(
            text=f"Dataset folder selected:\n{self.dataset_folder}"
        )

    def generate_eda_outputs(self):
        """Run Stage 1 EDA from the GUI."""

        # Inform the user that EDA generation has started
        self.result_label.config(
            text="Generating Stage 1 EDA outputs... please wait."
        )

        # Refresh the GUI immediately
        self.root.update()

        try:

            # Run the Stage 1 EDA workflow
            results = run_stage1_eda(self.dataset_folder, self.output_folder)

            # Get summary statistics from the EDA results
            summary = results["summary"]

            # Create summary text for display inside the GUI
            summary_text = (
                "Stage 1 EDA completed successfully.\n\n"
                f"Total images: {summary['total_images']}\n"
                f"Total classes: {summary['total_classes']}\n"
                f"Average width: {summary['average_width']}\n"
                f"Average height: {summary['average_height']}\n\n"
                "Generated EDA files:\n"
                "- dataset_index.csv\n"
                "- dataset_summary.csv\n"
                "- class_distribution.png\n"
                "- image_size_distribution.png\n"
                "- width_height_scatter.png\n"
                "- sample_image_grid.png\n"
                "- width_by_class_boxplot.png\n"
                "- height_by_class_boxplot.png\n"
                "- pixel_intensity_histogram.png\n"
                "- image_quality_issues.csv\n"
                "- class_imbalance_report.md\n"
                "- stage2_recommendations.md\n\n"
                f"Outputs saved to:\n{results['output_folder']}"
            )

            # Display EDA results in the GUI
            self.result_label.config(text=summary_text)

            # Display success pop-up message
            messagebox.showinfo(
                "EDA Complete",
                "Stage 1 EDA outputs generated successfully."
            )

        # Handle missing dataset folder errors
        except FileNotFoundError as error:
            messagebox.showerror("Dataset Not Found", str(error))
            self.result_label.config(text="Dataset folder was not found.")

        # Handle datasets with no valid images
        except ValueError as error:
            messagebox.showerror("No Images Found", str(error))
            self.result_label.config(text="No valid images were found.")

        # Handle all other unexpected errors
        except Exception as error:
            messagebox.showerror("EDA Error", str(error))
            self.result_label.config(text=f"EDA failed:\n{error}")

    def choose_image(self):
        """Allow the user to choose an image and display a preview."""

        # Open a file picker for image selection
        file_path = filedialog.askopenfilename(
            title="Choose a macroinvertebrate image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                ("All Files", "*.*")
            ]
        )

        # Stop if no image is selected
        if not file_path:
            return

        # Save the selected image path
        self.selected_image_path = Path(file_path)

        # Display the selected image
        self.display_image(self.selected_image_path)

        # Update GUI text
        self.result_label.config(
            text=f"Selected image:\n{self.selected_image_path.name}"
        )

    def display_image(self, image_path):
        """Display the selected image inside the GUI."""

        # Open image using Pillow
        image = Image.open(image_path)

        # Resize image so it fits nicely inside the GUI
        image.thumbnail((600, 400))

        # Convert image into a Tkinter-compatible format
        self.preview_image = ImageTk.PhotoImage(image)

        # Display image in the preview area
        self.image_label.config(image=self.preview_image, text="")
        self.image_label.image = self.preview_image

    def show_image_details(self):
        """Show basic image information."""

        # Make sure an image has been selected first
        if self.selected_image_path is None:
            messagebox.showwarning(
                "No Image Selected",
                "Please choose an image first."
            )
            return

        # Open image to access metadata
        image = Image.open(self.selected_image_path)

        # Create image information text
        details = (
            f"File name: {self.selected_image_path.name}\n"
            f"Image format: {image.format}\n"
            f"Width: {image.width} pixels\n"
            f"Height: {image.height} pixels\n"
            f"File location: {self.selected_image_path}"
        )

        # Display image details in the GUI
        self.result_label.config(text=details)

    def show_output_folder(self):
        """Show where the EDA outputs are saved."""

        # Check if EDA outputs have been generated yet
        if not self.output_folder.exists():
            messagebox.showwarning(
                "Output Folder Not Found",
                "Generate Stage 1 EDA outputs first."
            )
            return

        # Display the EDA output folder location
        self.result_label.config(
            text=(
                "EDA outputs are saved here:\n"
                f"{self.output_folder}\n\n"
                "Generated files include:\n"
                "- dataset_index.csv\n"
                "- dataset_summary.csv\n"
                "- class_distribution.png\n"
                "- image_size_distribution.png\n"
                "- width_height_scatter.png\n"
                "- sample_image_grid.png\n"
                "- width_by_class_boxplot.png\n"
                "- height_by_class_boxplot.png\n"
                "- pixel_intensity_histogram.png\n"
                "- image_quality_issues.csv\n"
                "- class_imbalance_report.md\n"
                "- stage2_recommendations.md"
            )
        )


def main():
    """Start the Tkinter GUI application."""

    # Create the Tkinter root window
    root = tk.Tk()

    # Create the GUI application object
    app = MacroGuiMenu(root)

    # Start the Tkinter event loop
    root.mainloop()


# Only run main if this file is executed directly
if __name__ == "__main__":
    main()
