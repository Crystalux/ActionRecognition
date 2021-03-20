import tensorflow_datasets as tfds

class CreateDataset(tfds.core.GeneratorBasedBuilder):

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(256, 256, 3)),
                'label': tfds.features.ClassLabel(names=['no', 'yes']),
            }),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Download the data and define splits."""
        extracted_path = dl_manager.download_and_extract('http://data.org/data.zip')
        # dl_manager returns pathlib-like objects with `path.read_text()`,
        # `path.iterdir()`,...
        return {
            'train': self._generate_examples(path=extracted_path / 'train_images'),
            'test': self._generate_examples(path=extracted_path / 'test_images'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[Key, Example]]:
        """Generator of examples for each split."""
        for img_path in path.glob('*.jpg'):
            # Yields (key, example)
            yield img_path.name, {
                'image': img_path,
                'label': 'yes' if img_path.name.startswith('yes_') else 'no',
            }
