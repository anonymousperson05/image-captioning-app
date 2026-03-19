def load_captions(path):
    captions = []
    mapping = {}

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if i == 0:
                continue  # skip header: image,caption

            parts = line.strip().split(',', 1)  # split only first comma

            if len(parts) < 2:
                continue

            img = parts[0]
            caption = parts[1]

            if img not in mapping:
                mapping[img] = []

            mapping[img].append(caption)
            captions.append(caption)

    return mapping, captions