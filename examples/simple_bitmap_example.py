import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import time

from src.fit_to_points_sequence import FitterToPointsSequence
from src.sequence_segment import print_segments_info


# Step 1: Create a simple test bitmap (or load your own image)
def create_sample_bitmap():
    """Create a small bitmap with a simple shape"""
    bitmap = np.zeros((100, 100), dtype=np.uint8)
    bitmap[20:80, 30:70] = 1
    # Add some features to make it more interesting
    bitmap[30:50, 50:80] = 1
    return bitmap


# Step 2: Show the bitmap
def show_bitmap(bitmap):
    """Display the original bitmap"""
    plt.figure(figsize=(8, 8))
    plt.imshow(bitmap, cmap='gray')
    plt.title('Step 1: Original Bitmap', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('step1_bitmap.png', dpi=150, bbox_inches='tight')
    print("Step 1 saved to 'step1_bitmap.png'")
    plt.show()


# Step 3: Extract contours
def extract_contours(bitmap):
    """Extract contours from a bitmap using scikit-image"""
    # find_contours returns list of (n,2) arrays of (row, col) coordinates
    contours = measure.find_contours(bitmap, level=0.5)

    if len(contours) == 0:
        print("No contours found!")
        return None

    # Use the first (or largest) contour
    contour = contours[0]
    print(f"Found contour with {len(contour)} points")
    return contour


# Step 4: Show the contour
def show_contour(bitmap, contour):
    """Display the extracted contour"""
    plt.figure(figsize=(8, 8))
    plt.imshow(bitmap, cmap='gray', alpha=0.3)
    plt.plot(contour[:, 1], contour[:, 0], 'b-', linewidth=2, label='Contour')
    plt.title(f'Step 2: Contour from skimage ({len(contour)} points)',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('step2_contour.png', dpi=150, bbox_inches='tight')
    print("Step 2 saved to 'step2_contour.png'")
    plt.show()


# Step 5: Fit polygon using FitterToPointsSequence
def fit_polygon(contour, verbose = True):
    """
    Fit line segments to contour using FitterToPointsSequence.

    Args:
        contour: numpy array of contour points
        is_closed: Whether the contour is closed
        max_segments: Maximum number of segments to fit
        tolerance: Fitting tolerance
        verbose: Print fitting progress

    Returns:
        List of SequenceSegment objects
    """
    # Create fitter instance and measure fitting time
    start_time = time.perf_counter()

    fitter = FitterToPointsSequence(
        points_sequence=contour,
        is_closed=True,
        max_segments_count=15,
        max_adjust_iterations=20,
        tolerance=0.2,
        #verbose=verbose
        verbose=False
    )

    # Fit line segments to the contour
    segments = fitter.fit()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"\nFitted {len(segments)} segments to the contour")
    print(f"Fitting time: {elapsed_time*1000:.2f} ms ({elapsed_time:.4f} seconds)")
    return segments


# Step 6: Show the fitted polygon
def show_fitted_polygon(bitmap, contour, segments):
    """Display the fitted line segments"""
    plt.figure(figsize=(10, 8))
    plt.imshow(bitmap, cmap='gray', alpha=0.3)
    plt.plot(contour[:, 1], contour[:, 0], 'k.', alpha=0.2, markersize=3,
             label='Original contour points')

    # Draw each segment
    colors = plt.cm.rainbow(np.linspace(0, 1, len(segments)))
    for i, segment in enumerate(segments):
        # Get the points for this segment
        if segment.first_index <= segment.last_index:
            segment_points = segment.whole_sequence[segment.first_index:segment.last_index+1]
        else:  # Handle circular/closed case
            segment_points = np.vstack([
                segment.whole_sequence[segment.first_index:],
                segment.whole_sequence[:segment.last_index+1]
            ])

        # Plot segment points and line
        plt.plot(segment_points[:, 1], segment_points[:, 0], 'o-',
                color=colors[i], linewidth=2.5, markersize=5,
                label=f'Seg {i} ({segment.points_count()} pts)')

    plt.title(f'Step 3: FitterToPointsSequence Result ({len(segments)} segments)',
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('step3_fitted_segments.png', dpi=150, bbox_inches='tight')
    print("Step 3 saved to 'step3_fitted_segments.png'")
    plt.show()


# Example usage - demonstrating all 6 modular steps
if __name__ == "__main__":
    # Step 1: Create bitmap
    print("=" * 60)
    print("STEP 1: Creating bitmap")
    print("=" * 60)
    bitmap = create_sample_bitmap()

    # Step 2: Show bitmap
    print("\n" + "=" * 60)
    print("STEP 2: Showing bitmap")
    print("=" * 60)
    show_bitmap(bitmap)

    # Step 3: Extract contours
    print("\n" + "=" * 60)
    print("STEP 3: Extracting contours")
    print("=" * 60)
    contour = extract_contours(bitmap)

    # Step 4: Show contour
    print("\n" + "=" * 60)
    print("STEP 4: Showing contour")
    print("=" * 60)
    show_contour(bitmap, contour)

    # Step 5: Fit polygon
    print("\n" + "=" * 60)
    print("STEP 5: Fitting polygon with FitterToPointsSequence")
    print("=" * 60)
    segments = fit_polygon(contour)

    print_segments_info(segments)
    show_fitted_polygon(bitmap, contour, segments)

    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETED!")
    print("=" * 60)