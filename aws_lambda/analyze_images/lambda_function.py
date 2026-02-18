import base64
import json
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps


def _response(status_code: int, body: dict):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Authorization,Content-Type",
            "Access-Control-Allow-Methods": "OPTIONS,POST",
        },
        "body": json.dumps(body),
    }


def _extract_api_key(headers: dict):
    if not headers:
        return None
    auth = headers.get("authorization") or headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        return None
    return auth.split(" ", 1)[1].strip()


def _download_single_image(index: int, url: str, total_count: int):
    print(f"Downloading image {index + 1}/{total_count}")

    response = requests.get(url, timeout=180, stream=True)
    response.raise_for_status()

    image_bytes = response.content
    if not image_bytes:
        raise ValueError(f"Downloaded image at index {index} is empty")

    content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
    if not content_type:
        parsed_url = urlparse(url)
        guessed_content_type, _ = mimetypes.guess_type(parsed_url.path)
        content_type = guessed_content_type or "image/jpeg"

    return {
        "index": index,
        "url": url,
        "image_bytes": image_bytes,
        "content_type": content_type,
    }


def _process_single_image(image_item: dict):
    index = image_item["index"]
    image_bytes = image_item["image_bytes"]
    content_type = image_item["content_type"]

    original_size = len(image_bytes)
    optimized_bytes = image_bytes
    optimized_content_type = content_type

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img = ImageOps.exif_transpose(img)

            max_dimension = 1568
            if max(img.size) > max_dimension:
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

            output = BytesIO()
            has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)

            if has_alpha:
                img.save(output, format="PNG", optimize=True, compress_level=9)
                optimized_content_type = "image/png"
            else:
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                img.save(output, format="JPEG", quality=82, optimize=True, progressive=True)
                optimized_content_type = "image/jpeg"

            candidate_bytes = output.getvalue()
            if candidate_bytes and len(candidate_bytes) < len(image_bytes):
                optimized_bytes = candidate_bytes
    except Exception as err:
        print(f"Optimization failed at index {index}, using original bytes: {str(err)}")

    optimized_size = len(optimized_bytes)
    encoded_image = base64.b64encode(optimized_bytes).decode("utf-8")
    data_url = f"data:{optimized_content_type};base64,{encoded_image}"

    return {
        "index": index,
        "original_size": original_size,
        "optimized_size": optimized_size,
        "label": {"type": "text", "text": f"Image Index {index}:"},
        "image": {"type": "image_url", "image_url": {"url": data_url}},
    }


def lambda_handler(event, context):
    if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
        return _response(200, {"ok": True})

    request_started_at = time.perf_counter()

    try:
        api_key = _extract_api_key(event.get("headers", {}))
        if not api_key:
            return _response(401, {"detail": "Authorization header with Bearer token is required"})

        raw_body = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            raw_body = base64.b64decode(raw_body).decode("utf-8")
        payload = json.loads(raw_body)

        prompt = payload.get("prompt", "")
        image_urls = payload.get("image_urls", "")
        model = payload.get("model", "")
        reasoning_effort = payload.get("reasoning_effort", "none")

        if not prompt:
            return _response(400, {"detail": "prompt is required"})
        if not image_urls:
            return _response(400, {"detail": "image_urls is required"})
        if not model:
            return _response(400, {"detail": "model is required"})

        urls = [u.strip().replace(" ", "%20") for u in image_urls.split(",") if u.strip()]
        if len(urls) == 0:
            return _response(400, {"detail": "image_urls must contain at least one valid URL"})

        content_array = [{"type": "text", "text": prompt}]

        # Stage 1: Download images in parallel
        download_started_at = time.perf_counter()
        download_workers = min(8, len(urls))
        downloaded_images = {}
        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            future_map = {
                executor.submit(_download_single_image, index, url, len(urls)): index
                for index, url in enumerate(urls)
            }
            for future in as_completed(future_map):
                result = future.result()
                downloaded_images[result["index"]] = result
        download_elapsed_seconds = time.perf_counter() - download_started_at

        # Stage 2: Process/optimize images in parallel
        processing_started_at = time.perf_counter()
        process_workers = min(6, len(urls))
        processed_images = {}
        with ThreadPoolExecutor(max_workers=process_workers) as executor:
            process_future_map = {
                executor.submit(_process_single_image, item): item["index"]
                for item in downloaded_images.values()
            }
            for future in as_completed(process_future_map):
                result = future.result()
                processed_images[result["index"]] = result
        processing_elapsed_seconds = time.perf_counter() - processing_started_at

        total_original_bytes = 0
        total_optimized_bytes = 0
        for index in range(len(urls)):
            processed = processed_images[index]
            total_original_bytes += processed["original_size"]
            total_optimized_bytes += processed["optimized_size"]
            content_array.append(processed["label"])
            content_array.append(processed["image"])

        normalized_reasoning_effort = reasoning_effort if reasoning_effort != "" else "none"

        openai_payload = {
            "model": model,
            "messages": [{"role": "user", "content": content_array}],
            "reasoning_effort": normalized_reasoning_effort,
        }
        openai_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Stage 3: OpenAI request
        openai_started_at = time.perf_counter()
        openai_timeout_seconds = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "600"))
        openai_response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=openai_headers,
            json=openai_payload,
            timeout=openai_timeout_seconds,
        )
        openai_elapsed_seconds = time.perf_counter() - openai_started_at

        if openai_response.status_code != 200:
            return _response(
                openai_response.status_code,
                {"detail": f"OpenAI API error: {openai_response.text}"},
            )

        response_json = openai_response.json()
        analysis_text = ""
        if response_json.get("choices"):
            analysis_text = response_json["choices"][0].get("message", {}).get("content", "")

        total_elapsed_seconds = time.perf_counter() - request_started_at
        print(
            f"Analyze images timing - total: {total_elapsed_seconds:.2f}s, "
            f"download: {download_elapsed_seconds:.2f}s, "
            f"processing: {processing_elapsed_seconds:.2f}s, "
            f"openai: {openai_elapsed_seconds:.2f}s, "
            f"image_count: {len(urls)}"
        )

        return _response(
            200,
            {
                "success": True,
                "message": "Image analysis completed successfully",
                "image_count": len(urls),
                "model": model,
                "reasoning_effort": normalized_reasoning_effort,
                "analysis": analysis_text,
                "raw_response": response_json,
                "payload_bytes": {
                    "original_total": total_original_bytes,
                    "optimized_total": total_optimized_bytes,
                    "saved_bytes": max(total_original_bytes - total_optimized_bytes, 0),
                },
                "timings_seconds": {
                    "download": round(download_elapsed_seconds, 3),
                    "processing": round(processing_elapsed_seconds, 3),
                    "openai": round(openai_elapsed_seconds, 3),
                    "total": round(total_elapsed_seconds, 3),
                },
            },
        )
    except requests.exceptions.RequestException as err:
        return _response(400, {"detail": f"Request error: {str(err)}"})
    except Exception as err:
        return _response(500, {"detail": f"Failed to analyze images: {str(err)}"})
