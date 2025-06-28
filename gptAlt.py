from openai import OpenAI
import os
import base64
client = OpenAI(api_key=os.environ['GPTyuval'])


def gpt_alt_text(image_path, model='gpt-4o-mini', prompt='Generate alt test for this JWST image'):
    """
    Generate alt text for an image using OpenAI's GPT model.
    
    Args:
        image_path (str): Path to the image file.
        model (str): The OpenAI model to use for generating alt text.
        
    Returns:
        str: Generated alt text for the image.
    """
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text":prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
                    },
                },
            ]}
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Example usage
    image_path = "/home/innereye/Pictures/uranus.jpg"
    alt_text = gpt_alt_text(image_path)
    print(f"Generated alt text:\n{alt_text}")
# with open("/home/innereye/Pictures/uranus.jpg", "rb") as image_file:
#     image_bytes = image_file.read()

# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "user", "content": [
#             {"type": "text", "text": "What's in this image?"},
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
#                 },
#             },
#         ]}
#     ],
#     max_tokens=300,
# )
# print(response.choices[0].message.content)



# response = client.responses.create(
#   model="gpt-4.1",
#   input=[],
#   text={
#     "format": {
#       "type": "text"
#     }
#   },
#   reasoning={},
#   tools=[],
#   temperature=1,
#   max_output_tokens=2048,
#   top_p=1,
#   store=True
# )

# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "What’s in this image?"},
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
#                         "detail": "high"
#                     },
#                 },
#             ],
#         }
#     ],
#     max_tokens=300,
# )
# print(response.choices[0].message.content)

# completion = client.chat.completions.create(
#   model="gpt-4o-mini",
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant. Help me with my math homework!"}, # <-- This is the system message that provides context to the model
#     {"role": "user", "content": "Hello! Could you solve 2+2?"}  # <-- This is the user message for which the model will generate a response
#   ]
# )

# print("Assistant: " + completion.choices[0].message.content)

# model = "gpt-4o"
# url = 'https://cdn.bsky.app/img/feed_fullsize/plain/did:plc:sehcpqiv6horvxo5p4plxf3g/bafkreietlcabhothny5f4zqiux463evg4okekvji6446s6l33p36fed2ma@jpeg'
# response = client.chat.completions.create(
#     model=model,
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "Alt text for the image, JWST NIRCam picture of NGC 4214"},
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": url,
#                         "detail": "high"
#                     },
#                 },
#             ],
#         }
#     ],
#     max_tokens=300,
# )
# print(response.choices[0].message.content)

