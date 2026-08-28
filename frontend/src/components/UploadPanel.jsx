import React, { useCallback, useRef, useState } from "react";
import { UploadCloud, Image as ImageIcon } from "lucide-react";
import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";

export default function UploadPanel({ preview, status, onFileSelected }) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const handleFiles = useCallback(
    (files) => {
      const file = files?.[0];
      if (!file) return;
      onFileSelected(file);
    },
    [onFileSelected]
  );

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragOver(true);
    } else if (e.type === "dragleave") {
      setDragOver(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    if (e.dataTransfer.files) {
      handleFiles(e.dataTransfer.files);
    }
  };

  return (
    <Card
      className={`border-2 border-dashed transition-all duration-200 cursor-pointer overflow-hidden relative ${dragOver
        ? "border-primary bg-primary/5 scale-[1.01]"
        : "border-muted-foreground/20 hover:border-primary/50 hover:bg-accent/50"
        }`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => handleFiles(e.target.files)}
      />
      <CardContent className="flex flex-col items-center justify-center p-12 text-center relative min-h-[300px]">
        {preview ? (
          <div className="absolute inset-0 bg-black/5 flex flex-col items-center justify-center">
            <img
              src={preview}
              alt="Upload preview background"
              className="absolute inset-0 w-full h-full object-cover opacity-30 blur-md pointer-events-none"
            />
            <img
              src={preview}
              alt="Upload preview"
              className="max-h-[85%] max-w-[85%] rounded-md shadow-lg z-10 transition-transform duration-300 pointer-events-none"
            />
            {status === "loading" && (
              <div className="absolute inset-0 bg-background/80 z-20 flex flex-col items-center justify-center animate-in fade-in duration-300">
                <div className="flex space-x-2 mb-4">
                  <div className="w-3 h-3 bg-primary rounded-full animate-bounce"></div>
                  <div className="w-3 h-3 bg-primary rounded-full animate-bounce delay-75"></div>
                  <div className="w-3 h-3 bg-primary rounded-full animate-bounce delay-150"></div>
                </div>
                <span className="text-sm font-medium animate-pulse">
                  Analyzing image...
                </span>
              </div>
            )}

            <div className="absolute top-4 right-4 z-20 flex gap-2">
              <Button
                variant="secondary"
                size="sm"
                className="shadow-sm backdrop-blur-md bg-background/80 hover:bg-background"
                onClick={(e) => {
                  e.stopPropagation();
                  inputRef.current?.click();
                }}
              >
                <UploadCloud className="w-4 h-4 mr-2" />
                Upload Image
              </Button>
              <Button
                variant="destructive"
                size="sm"
                className="shadow-sm backdrop-blur-md w-9 px-0"
                onClick={(e) => {
                  e.stopPropagation();
                  if (onFileSelected) onFileSelected(null);
                }}
                title="Clear image"
              >
                <span className="font-bold">✕</span>
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4 animate-in zoom-in fade-in duration-500 text-muted-foreground">
            <div className="p-4 rounded-full bg-primary/10 text-primary">
              <UploadCloud className="w-10 h-10" />
            </div>
            <div className="space-y-2">
              <h3 className="text-xl font-semibold tracking-tight text-foreground">
                Drop your image here
              </h3>
              <p className="text-sm max-w-sm">
                Drag a file into this area, or click anywhere to open your file browser.
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs opacity-75 mt-4">
              <ImageIcon className="w-4 h-4" />
              <span>Supports JPEG, PNG, WEBP</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
