#!/usr/bin/env python3
import argparse
import os
import pysrt

def extract_text_from_srt(srt_file_path):
    """
    Extracts the text from an SRT subtitle file using the pysrt library.

    Args:
        srt_file_path (str): The path to the SRT file.

    Returns:
        str: The extracted text.
    """
    try:
        subs = pysrt.open(srt_file_path, encoding='utf-8')
        return subs.text
    except Exception as e:
        print(f"Error processing file {srt_file_path}: {e}")
        return None

def main():
    """
    Main function to parse arguments and process the SRT file.
    """
    parser = argparse.ArgumentParser(description='Extract text from an SRT file using pysrt.')
    parser.add_argument('input_file', help='The path to the input SRT file.')
    args = parser.parse_args()

    input_path = args.input_file
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    extracted_text = extract_text_from_srt(input_path)

    if extracted_text is not None:
        # Create the output file path
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}.txt"

        # Save the extracted text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)

        print(f"Extracted text saved to {output_path}")

if __name__ == '__main__':
    main()