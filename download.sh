# Get presigned URL and download file
RESPONSE=$(curl -X POST "https://mozilladatacollective.com/api/datasets/cmu5x5tj000f2nq0755h8qcy5/download" \
  -H "Authorization: Bearer XXX" \
  -H "Content-Type: application/json")

# Extract download URL and download file
DOWNLOAD_URL=$(echo $RESPONSE | jq -r '.downloadUrl')
curl -o "common-voice-scripted-speech-27-0-thai-3aa1b7de.tar.gz" "$DOWNLOAD_URL"