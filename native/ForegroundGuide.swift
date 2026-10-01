import Foundation
import Vision
import CoreImage

// A fast region guide only. BiRefNet still produces the final product matte.
let context=CIContext()
func run(_ input:String,_ output:String) throws {
    let handler=VNImageRequestHandler(url:URL(fileURLWithPath:input),options:[:])
    let request=VNGenerateForegroundInstanceMaskRequest()
    try handler.perform([request])
    guard let observation=request.results?.first else {
        throw NSError(domain:"BrickStudio",code:1,userInfo:[NSLocalizedDescriptionKey:"No foreground located"])
    }
    let buffer=try observation.generateScaledMaskForImage(forInstances:observation.allInstances,from:handler)
    try context.writePNGRepresentation(of:CIImage(cvPixelBuffer:buffer),to:URL(fileURLWithPath:output),format:.L8,colorSpace:CGColorSpaceCreateDeviceGray())
}
guard CommandLine.arguments.count == 3 else { exit(2) }
do { try run(CommandLine.arguments[1],CommandLine.arguments[2]) }
catch { fputs("\(error)\n",stderr);exit(1) }
