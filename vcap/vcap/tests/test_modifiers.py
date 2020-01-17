import numpy as np

from vcap import (
    DetectionNode,
    BoundingBox,
    rect_to_coords,
    Crop,
    Clamp,
    Resize,
    SizeFilter
)


def test_crop():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    cropped = Crop(30, 30, 40, 40).apply(frame)
    assert cropped.shape == (10, 10, 3)

    cropped = Crop(90, 90, 110, 110).apply(frame)
    assert cropped.shape == (9, 9, 3)

    cropped = Crop(-10, -10, 10, 10).apply(frame)
    assert cropped.shape == (10, 10, 3)

    cropped = Crop(0, 0, 100, 100).apply(frame)
    assert cropped.shape == (100, 100, 3)

    cropped = Crop(10, 10, 90, 90).apply(frame)
    assert cropped.shape == (80, 80, 3)

    cropped = Crop(10, 10, 20, 20).pad_percent(10, 10, 10, 10).apply(frame)
    assert cropped.shape == (12, 12, 3)

    cropped = Crop(10, 10, 20, 20).pad_px(10, 10, 10, 10).apply(frame)
    assert cropped.shape == (30, 30, 3)

    node = DetectionNode(name="person",
                         coords=rect_to_coords([10, 10, 20, 20]))
    cropped = Crop.from_detection(node).apply(frame)
    assert cropped.shape == (10, 10, 3)


def test_clamp():
    frame = np.zeros((800, 800, 3), dtype=np.uint8)
    clamp = Clamp(frame, 100, 100)
    assert clamp.apply().shape == (100, 100, 3)
    detection_node = DetectionNode(name="person",
                                   coords=rect_to_coords([10, 10, 100, 100]))
    clamp.scale_detection_nodes([detection_node])
    assert detection_node.bbox == BoundingBox(80, 80, 800, 800)

    frame = np.zeros((800, 600, 3), dtype=np.uint8)
    clamp = Clamp(frame, 100, 100)
    assert clamp.apply().shape == (100, 75, 3)
    detection_node = DetectionNode(name="person",
                                   coords=rect_to_coords([10, 10, 100, 100]))
    clamp.scale_detection_nodes([detection_node])
    assert detection_node.bbox == BoundingBox(80, 80, 800, 800)

    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    clamp = Clamp(frame, 100, 100)
    assert clamp.apply().shape == (75, 100, 3)
    detection_node = DetectionNode(name="person",
                                   coords=rect_to_coords([10, 10, 100, 100]))
    clamp.scale_detection_nodes([detection_node])
    assert detection_node.bbox == BoundingBox(80, 80, 800, 800)


def test_size_filter():
    node = DetectionNode(
        name="person",
        coords=rect_to_coords([10, 10, 20, 20]))

    assert len(SizeFilter([node])
               .apply()) == 1
    assert len(SizeFilter([node])
               .min_size(12, 12)
               .max_size(100, 100)
               .apply()) == 0
    assert len(SizeFilter([node])
               .min_size(5, 5)
               .max_size(8, 8)
               .apply()) == 0
    assert len(SizeFilter([node])
               .min_size(5, 5)
               .max_size(15, 15)
               .apply()) == 1
    assert len(SizeFilter([node])
               .min_area(10 * 10)
               .max_area(11 * 11)
               .apply()) == 1
    assert len(SizeFilter([node])
               .min_area(11 * 11)
               .max_area(11 * 11).apply()) == 0

    assert len(SizeFilter([node])
               .min_area(9 * 9)
               .max_area(9 * 9)
               .apply()) == 0


def test_resize():
    input_width, input_height = 5, 10
    frame = np.arange(50, dtype=np.uint8).reshape((input_height, input_width))

    # Basic resize up
    frame_resize = Resize(frame) \
        .resize(10, 20, Resize.ResizeType.FIT_BOTH).frame
    assert frame_resize.shape[1] == 10
    assert frame_resize.shape[0] == 20

    frame_resize = Resize(frame) \
        .resize(10, 20, Resize.ResizeType.FIT_ONE).frame
    assert frame_resize.shape[1] == 10
    assert frame_resize.shape[0] == 20

    # Resize up where target aspect ratio is wider than source
    frame_resize = Resize(frame) \
        .resize(30, 30, Resize.ResizeType.FIT_BOTH).frame
    assert frame_resize.shape[1] == 15
    assert frame_resize.shape[0] == 30

    frame_resize = Resize(frame) \
        .resize(30, 30, Resize.ResizeType.FIT_ONE).frame
    assert frame_resize.shape[1] == 30
    assert frame_resize.shape[0] == 60

    # Resize up where target aspect ratio is taller than source
    frame_resize = Resize(frame) \
        .resize(10, 30, Resize.ResizeType.FIT_BOTH).frame
    assert frame_resize.shape[1] == 10
    assert frame_resize.shape[0] == 20

    frame_resize = Resize(frame) \
        .resize(10, 30, Resize.ResizeType.FIT_ONE).frame
    assert frame_resize.shape[1] == 15
    assert frame_resize.shape[0] == 30

    # Resize to width
    frame_resize = Resize(frame).resize(30, -1, Resize.ResizeType.WIDTH).frame
    assert frame_resize.shape[1] == 30
    assert frame_resize.shape[0] == 60

    # Resize to height
    frame_resize = Resize(frame).resize(-1, 30, Resize.ResizeType.HEIGHT).frame
    assert frame_resize.shape[1] == 15
    assert frame_resize.shape[0] == 30

    # Resize exactly
    frame_resize = Resize(frame).resize(8, 7, Resize.ResizeType.EXACT).frame
    assert frame_resize.shape[1] == 8
    assert frame_resize.shape[0] == 7

    # Resize where the scaling is not an integer
    # Round up
    frame_resize = Resize(frame) \
        .resize(10, 15, Resize.ResizeType.FIT_BOTH).frame
    assert frame_resize.shape[1] == 8
    assert frame_resize.shape[0] == 15
    # Round down
    input_width, input_height = 15, 4
    frame = np.arange(60, dtype=np.uint8).reshape((input_height, input_width))
    frame_resize = Resize(frame) \
        .resize(20, 10, Resize.ResizeType.FIT_BOTH).frame
    assert frame_resize.shape[1] == 20
    assert frame_resize.shape[0] == 5


def test_resize_crop():
    input_width, input_height = 5, 10
    frame = np.arange(50, dtype=np.uint8).reshape((input_height, input_width))

    # Simple right/bottom crop
    frame_resize = Resize(frame) \
        .crop(3, 4, Resize.CropPadType.RIGHT_BOTTOM) \
        .frame
    # noinspection PyUnresolvedReferences
    assert (frame[:4, :3] == frame_resize).all()

    # Simple top/left crop
    frame_resize = Resize(frame).crop(4, 3, Resize.CropPadType.LEFT_TOP).frame
    # noinspection PyUnresolvedReferences
    assert (frame[-3:, -4:] == frame_resize).all()

    # Crop starting at a point
    frame_resize = Resize(frame) \
        .crop(2, 7, Resize.CropPadType.CROP_START_POINT, top_left=(1, 3)).frame
    # noinspection PyUnresolvedReferences
    assert (frame[3:10, 1:3] == frame_resize).all()

    # Crop all sides (keep center)
    frame_resize = Resize(frame).crop(2, 3, Resize.CropPadType.ALL).frame
    # noinspection PyUnresolvedReferences
    assert (frame[3:6, 1:3] == frame_resize).all()

    # Crop larger than frame should return frame
    frame_resize = Resize(frame).crop(6, 11, Resize.CropPadType.ALL).frame
    # noinspection PyUnresolvedReferences
    assert (frame == frame_resize).all()

    # CropPadType.NONE should be a nop
    frame_resize = Resize(frame).crop(-1, -1, Resize.CropPadType.NONE).frame
    # noinspection PyUnresolvedReferences
    assert (frame == frame_resize).all()


def test_resize_pad():
    input_width, input_height = 10, 5
    frame = np.arange(50, dtype=np.uint8).reshape((input_height, input_width))

    # Pad bottom/right, then top/left
    frame_resize = Resize(frame) \
        .pad(13, 9, 255, Resize.CropPadType.RIGHT_BOTTOM) \
        .pad(17, 11, 254, Resize.CropPadType.LEFT_TOP) \
        .frame
    frame_expected = np.pad(
        np.arange(50, dtype=np.uint8).reshape((input_height, input_width)),
        ((0, 4), (0, 3)),
        'constant',
        constant_values=255)
    frame_expected = np.pad(
        frame_expected,
        ((2, 0), (4, 0)),
        'constant',
        constant_values=254)
    assert frame_resize.shape[1] == 17
    assert frame_resize.shape[0] == 11
    assert (frame_resize == frame_expected).all()

    # Pad all around
    frame_resize = Resize(frame) \
        .pad(13, 9, 255, Resize.CropPadType.ALL) \
        .frame
    frame_expected = np.pad(
        np.arange(50, dtype=np.uint8).reshape((input_height, input_width)),
        ((2, 2), (1, 2)),
        'constant',
        constant_values=255)
    assert frame_resize.shape[1] == 13
    assert frame_resize.shape[0] == 9
    # noinspection PyUnresolvedReferences
    assert (frame_resize == frame_expected).all()

    # Crop larger than frame should return frame
    frame_resize = Resize(frame).pad(9, 4, -1, Resize.CropPadType.ALL).frame
    # noinspection PyUnresolvedReferences
    assert (frame == frame_resize).all()

    # CropPadType.NONE should be a nop
    frame_resize = Resize(frame).pad(-1, -1, -1, Resize.CropPadType.NONE).frame
    # noinspection PyUnresolvedReferences
    assert (frame == frame_resize).all()


def test_resize_scale():

    input_width, input_height = 5, 10
    frame = np.arange(50, dtype=np.uint8).reshape((input_height, input_width))

    # Single integer resize
    resize = Resize(frame).resize(10, 20, Resize.ResizeType.EXACT)
    node = DetectionNode(name="person",
                         coords=rect_to_coords([10, 10, 20, 20]))
    resize.scale_and_offset_detection_nodes([node])
    assert node.bbox == BoundingBox(5, 5, 10, 10)

    # Double integer resize (note that coords are rounded in node.bbox output)
    resize = Resize(frame) \
        .resize(10, 20, Resize.ResizeType.EXACT) \
        .resize(20, 40, Resize.ResizeType.EXACT)
    node = DetectionNode(name="person",
                         coords=rect_to_coords([15, 15, 20, 20]))
    resize.scale_and_offset_detection_nodes([node])
    assert node.bbox == BoundingBox(4, 4, 5, 5)

    # Single crop
    input_width, input_height = 20, 30
    frame = np.arange(600, dtype=np.uint8).reshape((input_height, input_width))
    resize = Resize(frame) \
        .crop(15, 20, Resize.CropPadType.LEFT_TOP)
    node = DetectionNode(name="person",
                         coords=rect_to_coords([15, 15, 20, 20]))
    resize.scale_and_offset_detection_nodes([node])
    assert node.bbox == BoundingBox(20, 25, 25, 30)

    # Two affecting crops plus one that should not change the offset
    input_width, input_height = 20, 30
    frame = np.arange(600, dtype=np.uint8).reshape((input_height, input_width))
    resize = Resize(frame) \
        .crop(15, 20, Resize.CropPadType.LEFT_TOP) \
        .crop(10, 15, Resize.CropPadType.RIGHT_BOTTOM) \
        .crop(8, 5, Resize.CropPadType.ALL)
    node = DetectionNode(name="person",
                         coords=rect_to_coords([15, 15, 20, 20]))
    resize.scale_and_offset_detection_nodes([node])
    assert node.bbox == BoundingBox(21, 30, 26, 35)

    # Crop then resize
    input_width, input_height = 20, 30
    frame = np.arange(600, dtype=np.uint8).reshape((input_height, input_width))
    resize = Resize(frame) \
        .crop(15, 20, Resize.CropPadType.LEFT_TOP) \
        .resize(30, 40, Resize.ResizeType.EXACT)
    node = DetectionNode(name="person",
                         coords=rect_to_coords([15, 15, 20, 20]))
    resize.scale_and_offset_detection_nodes([node])
    assert node.bbox == BoundingBox(13, 18, 15, 20)

    # Resize then crop
    input_width, input_height = 20, 30
    frame = np.arange(600, dtype=np.uint8).reshape((input_height, input_width))
    resize = Resize(frame) \
        .resize(30, 40, Resize.ResizeType.EXACT) \
        .crop(15, 20, Resize.CropPadType.LEFT_TOP)
    node = DetectionNode(name="person",
                         coords=rect_to_coords([15, 15, 20, 20]))
    resize.scale_and_offset_detection_nodes([node])
    assert node.bbox == BoundingBox(20, 26, 23, 30)
