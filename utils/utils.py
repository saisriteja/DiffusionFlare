import random
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF


def create_comparision_image(epoch, plot_list):
    selected_sample = random.choice(plot_list)  # select one sample randomly
    titles = ["RGB", "Depth", "Flare", "Output"]

    ind = random.randint(0, selected_sample[0].size(0) - 1)
    selected_samples = [tensor[ind] for tensor in selected_sample]
    pil_images = [
        TF.to_pil_image(tensor.squeeze(0).cpu()) for tensor in selected_samples
    ]
    width, height = pil_images[0].size
    total_w = width * 4
    total_h = height + 50
    main_title_text = f"Val_out_at_{epoch}"

    final_image = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(final_image)
    font = ImageFont.load_default()

    for i, img in enumerate(pil_images):
        final_image.paste(img, (i * width, 50))
        title_text = titles[i]
        text_bbox = draw.textbbox((0, 0), title_text, font=font)
        text_width, text_height = (
            text_bbox[2] - text_bbox[0],
            text_bbox[3] - text_bbox[1],
        )

        text_x = i * width + (width - text_width) // 2
        text_y = 25 - text_height // 2
        draw.text((text_x, text_y), title_text, fill="black", font=font)

    main_text_bbox = draw.textbbox((0, 0), main_title_text, font=font)
    main_text_width, _ = (
        main_text_bbox[2] - main_text_bbox[0],
        main_text_bbox[3] - main_text_bbox[1],
    )

    main_text_x = (total_w - main_text_width) // 2
    main_text_y = 10
    draw.text((main_text_x, main_text_y), main_title_text, fill="black", font=font)
    return final_image
