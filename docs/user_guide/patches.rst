Patch-Based Processing
======================

Large medical images (e.g., full-field mammograms at 4000x3000 pixels) may not fit on a GPU in one pass. The framework provides :class:`~medical_image.data.patch.PatchGrid` to split images into regular patches, process them independently, and reassemble the result.

Creating a Patch Grid
---------------------

.. code-block:: python

   from medical_image import PatchGrid

   grid = PatchGrid(image, patch_size=(128, 128))

   print(len(grid.patches))    # total number of patches
   print(len(grid.grid))       # number of rows
   print(len(grid.grid[0]))    # number of columns

If the image dimensions are not evenly divisible by the patch size, the grid **zero-pads** the bottom and right edges automatically.

Working with Individual Patches
-------------------------------

.. code-block:: python

   for patch in grid.patches:
       # Patch metadata
       row, col = patch.grid_id()          # grid position
       x, y = patch.pixel_position()       # pixel offset in original image
       print(patch.is_padded)              # True if this patch has padding

       # Convert to Image for processing
       patch_img = patch.to_image()
       output = patch_img.clone()

       # Process the patch
       algo(image=patch_img, output=output)

       # Write result back
       patch.pixel_data = output.pixel_data

Reconstructing the Full Image
------------------------------

After processing individual patches, reassemble the full image:

.. code-block:: python

   # As a tensor (padding removed)
   full_tensor = grid.reconstruct()  # shape matches original image

   # As an Image object
   result_image = grid.to_image()

Region of Interest
------------------

For targeted analysis, extract a sub-region using :class:`~medical_image.data.region_of_interest.RegionOfInterest`:

.. code-block:: python

   from medical_image import RegionOfInterest

   # From center coordinates
   roi = RegionOfInterest.from_center(image, cx=1250, cy=2000, half_size=127)
   roi_img = roi.load()

   # From bounding box
   roi = RegionOfInterest(image, coordinates=[100, 200, 356, 456])
   roi_img = roi.load()

   # Normalize 12-bit DICOM values
   normalized = RegionOfInterest.normalize(roi_img, divisor=4095.0)