"use client";

import { FIRResponse } from "./types";
import { Button } from "@/components/ui/button";
import { jsPDF } from "jspdf";

interface FIRDocumentTabProps {
  result: FIRResponse;
  copied: boolean;
  onCopy: () => void;
  onPrint: () => void;
}

export function FIRDocumentTab({ result, copied, onCopy, onPrint }: FIRDocumentTabProps) {
  const handleDownloadPDF = () => {
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 15;
    const contentWidth = pageWidth - margin * 2;
    let yPos = margin;

    // Set monospace font to match the <pre> display
    doc.setFont("courier", "normal");
    doc.setFontSize(9);
    doc.setTextColor(0, 0, 0);

    // Split the FIR text into lines
    const lines = result.fir_text.split('\n');

    lines.forEach((line: string) => {
      // Check if we need a new page
      if (yPos > pageHeight - margin) {
        doc.addPage();
        yPos = margin;
      }

      // Handle long lines by wrapping them
      if (line.length > 80) {
        const wrappedLines = doc.splitTextToSize(line, contentWidth);
        wrappedLines.forEach((wrappedLine: string) => {
          if (yPos > pageHeight - margin) {
            doc.addPage();
            yPos = margin;
          }
          doc.text(wrappedLine, margin, yPos);
          yPos += 4;
        });
      } else {
        doc.text(line, margin, yPos);
        yPos += 4;
      }
    });

    const fileName = `FIR_${result.fir_id}_${result.name.replace(/\s+/g, "_")}.pdf`;
    doc.save(fileName);
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="px-2.5 py-1 bg-primary/10 text-primary text-xs font-mono rounded-md border border-primary/20">
            {result.fir_id}
          </span>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={onCopy}
            className={`text-xs transition-all ${
              copied
                ? "bg-chart-2/10 text-chart-2 border-chart-2/20"
                : "hover:bg-chart-1/5 hover:text-chart-1 hover:border-chart-1/20"
            }`}
          >
            {copied ? "✓ Copied!" : "📋 Copy"}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleDownloadPDF}
            className="text-xs hover:bg-destructive/5 hover:text-destructive hover:border-destructive/20"
          >
            📄 PDF
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={onPrint}
            className="text-xs hover:bg-chart-3/5 hover:text-chart-3 hover:border-chart-3/20"
          >
            🖨 Print
          </Button>
        </div>
      </div>
      <div className="border border-border rounded-xl overflow-hidden">
        <div className="bg-muted/50 px-4 py-2 border-b border-border flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-destructive"></span>
          <span className="w-2 h-2 rounded-full bg-chart-4"></span>
          <span className="w-2 h-2 rounded-full bg-chart-2"></span>
          <span className="ml-2 text-xs text-muted-foreground">FIR Document</span>
        </div>
        <pre className="p-4 text-xs text-foreground font-mono whitespace-pre-wrap overflow-x-auto max-h-[500px] overflow-y-auto bg-card">
          {result.fir_text}
        </pre>
      </div>
    </div>
  );
}
