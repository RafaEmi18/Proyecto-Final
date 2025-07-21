import { useEffect, useRef, useState } from 'react';

export default function BrailleCamera() {
  const videoRef = useRef(null);
  const [letter, setLetter] = useState('');
  const [confidence, setConfidence] = useState(null);

  useEffect(() => {
    // Activar cámara
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    });
  }, []);

  const captureAndSend = async () => {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL('image/png').split(',')[1]; // base64

    const res = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData }),
    });

    const data = await res.json();
    setLetter(data.letter);
    setConfidence(data.confidence);
  };

  return (
    <div className="flex flex-col items-center gap-4">
      <video ref={videoRef} autoPlay width={640} height={480} className="border" />
      <button onClick={captureAndSend} className="px-4 py-2 bg-blue-600 text-white rounded">Detectar Letra</button>
      {letter && (
        <div className="mt-4 text-lg font-bold">
          Letra: {letter} <br />
          Confianza: {confidence?.toFixed(2)}
        </div>
      )}
    </div>
  );
}
