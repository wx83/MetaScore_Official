Save at: new_sample_electronic_140000_2
Pick some hard case where drum is generated with at least piano, I want it to be easy
                tgt_start[:, 0] = sos  # batch_size, sequence  # indexr[start-of-song]
                tgt_start[:, 1] = indexer['start-of-tags']
                tgt_start[:, 2] = indexer['start-of-genre']
                tgt_start[:, 3] = indexer['tag_genre_Electronic/Dance']