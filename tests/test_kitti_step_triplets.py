from types import SimpleNamespace

from predify2021.datasets.kitti_step_triplets import KITTISTEPTripletDataset


def test_triplets_are_consecutive_and_sequence_local():
    samples = [
        {"sequence_id": "0001", "frame_id": "000001", "image_path": "a"},
        {"sequence_id": "0001", "frame_id": "000000", "image_path": "b"},
        {"sequence_id": "0002", "frame_id": "000000", "image_path": "c"},
        {"sequence_id": "0001", "frame_id": "000002", "image_path": "d"},
        {"sequence_id": "0002", "frame_id": "000001", "image_path": "e"},
        {"sequence_id": "0002", "frame_id": "000002", "image_path": "f"},
    ]
    dataset = KITTISTEPTripletDataset(SimpleNamespace(samples=samples))

    assert len(dataset) == 2
    for sequence_id, triplet in dataset.triplets:
        assert {sample["sequence_id"] for sample in triplet} == {sequence_id}
        frame_ids = [int(sample["frame_id"]) for sample in triplet]
        assert frame_ids == list(range(frame_ids[0], frame_ids[0] + 3))
