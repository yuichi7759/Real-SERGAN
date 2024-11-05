import cv2
import numpy as np
import subprocess
import os

def split_image_into_tiles(image, tile_size, overlap):
    """
    入力画像をタイルに分割する。
    """
    tiles = []
    h, w = image.shape[:2]
    step = tile_size - overlap  # タイルのステップ幅（重なりを考慮）

    for y in range(0, h, step):
        for x in range(0, w, step):
            tile = image[y:y + tile_size, x:x + tile_size]
            print(x, y)

            # running exe file
            img_path = "tile.png"
            out_path = "tile_out.png"            
            cv2.imwrite(img_path, tile)
            #command = f"./realesrgan-ncnn-vulkan -i {img_path} -o {out_path} -n realesrgan-x4plus" # command
            #command = f"./realesrgan-ncnn-vulkan -i {img_path} -o {out_path} -n realesr-animevideov3-x2" # command      
            #command = f"./realesrgan-ncnn-vulkan -i {img_path} -o {out_path} -n realesrgan-x4plus-anime" # command      
            command = f"python inference_realesrgan.py -n RealESRGAN_x4plus -i {img_path} -o ./"                  
            
            subprocess.run(command, shell=True)

            processed_tile = cv2.imread(out_path)

            h_tile, w_tile = tile.shape[:2]
            processed_tile = cv2.resize(processed_tile, (w_tile, h_tile), cv2.INTER_CUBIC)

            #processed_tile = process_tile_with_realesrgan(tile_out, i)  # タイル処理
            tiles.append((x, y, processed_tile))
    return tiles

def alpha_blend_tiles(tiles, image_size, tile_size, overlap):
    """
    分割したタイルをアルファブレンディングして再合成する。
    """
    h, w = image_size
    blended_image = np.zeros((h, w, 3), dtype=np.float32)  # 合成用の画像
    alpha_map = np.zeros((h, w), dtype=np.float32)         # アルファブレンディング用のマスク

    # アルファブレンドのための重みマスク（グラデーション）
    mask = np.ones((tile_size, tile_size), dtype=np.float32)
    for i in range(overlap):
        mask[i, :] *= i / overlap
        mask[:, i] *= i / overlap
        mask[-i-1, :] *= i / overlap
        mask[:, -i-1] *= i / overlap

    # 各タイルをブレンディングしながら配置
    for x, y, tile in tiles:
        h_tile, w_tile = tile.shape[:2]

        # 合成する部分のアルファマスクとタイルのマスクをかける
        blended_image[y:y+h_tile, x:x+w_tile] += tile * mask[:h_tile, :w_tile, None]
        alpha_map[y:y+h_tile, x:x+w_tile] += mask[:h_tile, :w_tile]

    # 最終的に正規化して画像を取得
    blended_image /= np.maximum(alpha_map[:, :, None], 1e-6)  # ゼロ除算回避
    return np.clip(blended_image, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    # メイン処理
    #image_path = 'DJI_20241021132301_0043_D.JPG'  # 入力画像のパス
    image_path = 'DJI_20241021132303_0046_D.JPG'  # 入力画像のパス    
    tile_size = 1280  # タイルサイズ
    overlap = 32     # オーバーラップ量

    # 画像の読み込み
    image = cv2.imread(image_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCVはBGRなので、RGBに変換

    # タイル分割
    tiles = split_image_into_tiles(image, tile_size, overlap)

    # アルファブレンディングして合成
    blended_image = alpha_blend_tiles(tiles, image.shape[:2], tile_size, overlap)

    #dst = cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR)
    output_path = os.path.splitext(image_path)[0] + "_esrgan.png"
    cv2.imwrite(output_path, blended_image)

    # 結果を表示
    #cv2.imshow('Blended Image', blended_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
