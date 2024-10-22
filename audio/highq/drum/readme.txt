Save at: new_sample_withdurm2_140000_2
Pick some hard case where drum is generated with some other instruments. We want at least drum track is included
                tgt_start[:, 0] = sos  # batch_size, sequence  # indexr[start-of-song]
                tgt_start[:, 1] = indexer['start-of-tags']
                tgt_start[:, 2] = indexer['start-of-genre']
                tgt_start[:, 3] = indexer['tag_genre_None']
                tgt_start[:, 4] = indexer['start-of-composer']
                tgt_start[:, 5] = indexer['tag_composer_None']
                tgt_start[:, 6] = indexer['start-of-complexity']
                tgt_start[:, 7] = indexer['tag_complexity_None']
                tgt_start[:, 8] = indexer['start-of-instrument']
                tgt_start[:, 9] = indexer['tag_instrument_drums']