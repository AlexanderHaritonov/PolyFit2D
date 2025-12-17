import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from src.fit_to_points_sequence import FitterToPointsSequence

def show_fitted_polygon(bitmap, contour, segments):
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

    plt.title(f'({len(segments)} segments)',
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('fitted_segments.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    """Create a small bitmap with a simple shape"""
    bitmap = np.zeros((100, 100), dtype=np.uint8)
    bitmap[20:80, 30:70] = 1
    bitmap[30:50, 50:80] = 1

    """extract contour - a dense points sequence"""
    contour = measure.find_contours(bitmap, level=0.5)[0]

    """ fit a polygon """
    segments = FitterToPointsSequence(contour, is_closed=True).fit()

    """ plot """
    show_fitted_polygon(bitmap, contour, segments)
