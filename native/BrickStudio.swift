import SwiftUI
import AppKit
import UniformTypeIdentifiers
import ImageIO
import Darwin

struct ExportImages: Decodable { let jpeg:String; let png:String }
struct Photo: Identifiable, Decodable {
    let id:String
    let name:String
    let source:String
    let jpeg:String
    let png:String
    let before:String
    let width:Int
    let height:Int
    let review:[String]
    let bbox:[Int]?
    let seconds:Double?
    var variants:[String:ExportImages]? = nil
    var source_size:[Int]? = nil
}
struct Manifest: Decodable { let records:[Photo] }


struct BatchRecipe: Codable, Equatable {
    var exposure=0.65,contrast=0.0,warmth=0.25,fill=0.84,sharpness=0.75,denoise=0.0,shadow=2.0
    var aspect="square",framing="centered",mask_mode="efficient",shadow_method="classic"
    var size=2400
    var package_cleanup:Bool?=nil
    // "independent": each centred photo fills on its own. nil keeps one shared scale.
    var centered_scale:String?=nil
    // Whites: -1 brings the brightest tones down. nil or 0 leaves the photo as it was.
    // Optional so recipes saved before this control still decode.
    var whites:Double?=nil
    // Look harder for small loose pieces. nil keeps the normal sensitivity, so
    // recipes saved before this control still decode.
    var small_parts:Bool?=nil
    // Subject separation engine. nil/"mps" = Metal GPU (default); "onnx" = rembg CPU.
    // Optional so recipes saved before this control still decode.
    var mask_backend:String?=nil
    var shadow_detection:String?=nil
    var recover_parts:Bool?=nil
    var recover_white:Bool?=nil
    var recover_dark:Bool?=nil
    static func newFolder(_ base:BatchRecipe=BatchRecipe())->BatchRecipe {
        var recipe=base;recipe.shadow=1.15;recipe.shadow_method="local";recipe.denoise=1
        return recipe
    }
}
struct LearningExample:Codable,Equatable { var source:String;var distance:Double;var field:String }
struct PhotoAdjustment:Codable,Equatable {
    var values:[String:Double]
    var options:[String:String]?=nil
    var manually_edited:Bool?=nil
    var manual_fields:[String]=[]
    var confidence:String="similar"
    var model_version:Int=1
    var model_revision:String?=nil
    var examples:[LearningExample]?=nil
    var source_size:Int64?=nil
    var source_mtime:Double?=nil
}
struct LearningResponse:Decodable {
    var message:String?=nil
    var adjustments:[String:PhotoAdjustment]
    var errors:[LearningFailure]
    struct LearningFailure:Decodable { var source:String;var error:String }
}
struct PackagingLibrary:Codable {
    var recipe=BatchRecipe(package_cleanup:true)
    var marked:Set<String>=[]
    var overrides:[String:BatchRecipe]=[:]
}
struct BatchJob: Codable, Identifiable {
    var id=UUID().uuidString
    var name:String
    var folder:String
    var sources:[String]
    var override:BatchRecipe?=nil
    var started_recipe:BatchRecipe?=nil
    var photo_recipes:[String:BatchRecipe]?=nil
    var adjustment_recipes:[String:PhotoAdjustment]?=nil
    var status="queued"
    var output:String?=nil,metadata:String?=nil,message:String?=nil
    var completed=0,errors=0,attempts=0
    var include:Bool?=nil
    var archived:Bool?=nil
    // When true, the next queue run overwrites this job's existing export folder.
    var replace:Bool?=nil
}
struct BatchQueueState: Codable {
    var version=1
    var jobs:[BatchJob]=[]
    var shared_recipe=BatchRecipe.newFolder()
    var output_root=""
    var status="idle"
    var worker_pid:Int?=nil,child_pid:Int?=nil
    var progress:Double?=nil,elapsed_seconds:Double?=nil,eta_seconds:Double?=nil
    var active_name:String?=nil,phase:String?=nil,summary:String?=nil
    var active_job_id:String?=nil,photo_stage:String?=nil
    var photo_total:Int?=nil
    var completed_photos:Int?=nil,total_photos:Int?=nil,completed_batches:Int?=nil,error_count:Int?=nil
    var photo_number:Int?=nil,photo_elapsed:Double?=nil,photo_progress:Double?=nil
    var seconds_per_photo:Double?=nil,last_render_seconds:Double?=nil
    // Wall-clock span of the latest Process run (seconds since 1970); the worker keeps these keys.
    var run_started:Double?=nil,run_ended:Double?=nil
}

@MainActor final class Studio:ObservableObject {
    let root:URL
    let previewSessionID=UUID().uuidString
    var previewDirectory:URL { root.appendingPathComponent("work/live-preview/session-\(previewSessionID)") }
    @Published var queue=BatchQueueState()
    @Published var queueVisible=true
    @Published var foldersOnLeft=UserDefaults.standard.bool(forKey:"foldersOnLeft") {
        didSet { UserDefaults.standard.set(foldersOnLeft,forKey:"foldersOnLeft") }
    }
    @Published var recipeTarget:String?
    @Published var completedVisible=false
    /// Which photo of each folder the folder view shows (index into its sources); session only.
    @Published var batchAngles:[String:Int]=[:]
    @Published var foldersHidden=false
    /// True once a queue run has been seen this session, so a stale saved status never shows a footer.
    @Published var runSeen=false
    /// Pause was pressed; the worker stops after its current write.
    @Published var pauseRequested=false
    @Published var learningNoteFresh=false
    var learningFlashStamp=UUID()
    @Published var batchDropTarget:String?
    var batchSelectionAnchor:String?
    var draggedBatchIDs:[String]=[]
    static let batchDragType=UTType.utf8PlainText.identifier
    var activeBatches:[BatchJob] { queue.jobs.filter { $0.archived != true } }
    var completedBatches:[BatchJob] { queue.jobs.filter { $0.archived == true } }
    @Published var selectedBatchID:String?
    @Published var previewFolderID:String?
    @Published var batchDetailImage:NSImage?
    @Published var batchDetailSignature=""
    var batchDetailJobID:String?
    var batchPreviewDetailSerial:Int?
    var selectedBatch:BatchJob? { activeBatches.first{$0.id==selectedBatchID} }
    var recipeLocked:Bool { queue.jobs.first{$0.id==recipeTarget}?.started_recipe != nil }
    @Published var batchImages:[String:NSImage]=[:]
    @Published var batchPreviewNotes:[String:String]=[:]
    var batchPreviewSignatures:[String:String]=[:]
    var batchPreviewActive:(id:String,signature:String,serial:Int,isProbe:Bool)?
    var batchPreviewFailures:[String:String]=[:]
    var batchProbeFailures:[String:String]=[:]
    var folderRecipes:[String:BatchRecipe]=[:]
    var folderRecipesAvailable=true
    var folderRecipesURL:URL {
        FileManager.default.urls(for:.applicationSupportDirectory,in:.userDomainMask)[0]
            .appendingPathComponent("Brick Studio/folder-recipes.json")
    }
    @Published var packaging=PackagingLibrary()
    @Published var editingPackagePath:String?
    var regularRecipe=BatchRecipe()
    var switchingPhotoRecipe=false
    var packagingAvailable=true
    var packagingURL:URL {
        FileManager.default.urls(for:.applicationSupportDirectory,in:.userDomainMask)[0]
            .appendingPathComponent("Brick Studio/packaging-photos.json")
    }
    var normalRecipe:BatchRecipe { editingPackagePath == nil && editingAdjustmentPath == nil ? batchRecipe:regularRecipe }
    @Published var editingAdjustmentPath:String?
    @Published var photoAdjustments:[String:PhotoAdjustment]=[:]
    @Published var learningBusy=false
    @Published var learningNote="Learns from your exported corrections."
    @Published var autoLearn = UserDefaults.standard.object(forKey:"autoLearnRecipes") as? Bool ?? true {
        didSet { UserDefaults.standard.set(autoLearn,forKey:"autoLearnRecipes") }
    }
    var learningProcess:Process?
    var learningTimer:Timer?
    var learningCancelled=false
    var learningApplied:Set<String>=[]
    var learningPending:Set<String>=[]
    var learningRevisions:[String:Int]=[:]
    var adjustmentsAvailable=true
    var adjustmentsURL:URL { root.appendingPathComponent("work/learning/photo-adjustments.json") }
    var applyingRecipe=false
    /// The recipe as it was loaded into the panel, so an edit can be told from a selection change.
    var appliedRecipe:BatchRecipe?=nil
    var queueProcess:Process?
    var queueTimer:Timer?
    var queueURL:URL { root.appendingPathComponent("work/queue/state.json") }
    var queueRunning:Bool { queue.status == "running" || queue.status == "starting" }
    @Published var inputs:[URL]=[]
    @Published var photos:[Photo]=[]
    @Published var included:Set<String>=[] { didSet { schedulePreview() } }
    @Published var selected=0 { didSet { activatePhotoRecipe();schedulePreview() } }
    @Published var mode="Edited"
    @Published var exposure=0.65 { didSet { schedulePreview() } }
    @Published var contrast=0.0 { didSet { schedulePreview() } }
    @Published var whites=0.0 { didSet { schedulePreview() } }
    @Published var warmth=0.25 { didSet { schedulePreview() } }
    @Published var fill=0.84 { didSet { schedulePreview() } }
    @Published var sharpness=0.75 { didSet { schedulePreview() } }
    @Published var denoise=0.0 { didSet { schedulePreview() } }
    @Published var packageCleanup=false { didSet { schedulePreview() } }
    @Published var enhancedShadow=false { didSet { schedulePreview() } }
    @Published var softShadowDetection=false { didSet { schedulePreview() } }
    @Published var shadow=2.0 { didSet { schedulePreview() } }
    @Published var aspect="square" { didSet { schedulePreview() } }
    @Published var exportFraming="centered" { didSet { schedulePreview() } }
    @Published var independentScale=false { didSet { schedulePreview() } }
    @Published var smallParts=false { didSet { if oldValue != smallParts { knownFrames=[:] }; schedulePreview() } }
    @Published var recoverParts=false { didSet { if oldValue != recoverParts { knownFrames=[:] }; schedulePreview() } }
    @Published var recoverWhite=false { didSet { if oldValue != recoverWhite { knownFrames=[:] }; schedulePreview() } }
    @Published var recoverDark=false { didSet { if oldValue != recoverDark { knownFrames=[:] }; schedulePreview() } }
    @Published var previewFraming="centered" { didSet { schedulePreview() } }
    var cropMode:String { exportFraming == "both" ? previewFraming : exportFraming }
    @Published var size=2400 { didSet { schedulePreview() } }
    @Published var maskMode="efficient" { didSet { if oldValue != maskMode { knownFrames=[:] }; schedulePreview() } }
    @Published var maskBackend="mps" { didSet { if oldValue != maskBackend { knownFrames=[:] }; schedulePreview() } }
    @Published var running=false
    @Published var progress=0.0
    @Published var photoProgress=0.0
    @Published var photoElapsed=0.0
    @Published var batchElapsed=0.0
    var queuePhotoKey="",queuePhotoStart:Date?=nil
    @Published var eta=0.0
    @Published var batchTotal=0
    @Published var batchCompleted=0
    @Published var activePhoto=0
    @Published var activeName=""
    @Published var phase=""
    @Published var status="Ready for your next shoot"
    @Published var error:String?
    @Published var output:URL
    @Published var previewPath:String?
    @Published var previewOriginal:String?
    @Published var previewSource=""
    @Published var previewLoading=false
    @Published var previewNote="Import photos to begin"
    @Published var previewSeconds=0.0
    @Published var previewReviews:[String:[String]]=[:]
    var exportSignatures:[String:String]=[:]
    var activeExportSignature=""
    var knownFrames:[String:[String:Any]]=[:]
    var metadata:URL
    var process:Process?
    var timer:Timer?
    var logRead:FileHandle?
    var logBuffer=Data()
    var outputParent:URL
    var batchStarted:Date?
    var photoStarted:Date?
    var etaReceived:Date?
    var etaValue=0.0
    var finishing=false
    var previewProcess:Process?
    var previewInput:FileHandle?
    var previewOutput:FileHandle?
    var previewBuffer=Data()
    var previewSerial=0
    var previewGeneration=0
    var previewWork:DispatchWorkItem?
    var previewEnabled=false
    let extensions=Set(["dng","cr2","cr3","nef","arw","raf","rw2","orf","pef","jpg","jpeg","png","tif","tiff"])
    var selectedPath:String? { inputs.indices.contains(selected) ? inputs[selected].path : nil }
    var checked:[URL] { inputs.filter { included.contains($0.path) } }
    var current:Photo? { photos.first { $0.source==selectedPath } }
    var currentReview:[String] {
        if matchesExport { return current?.review ?? [] }
        return previewReviews[selectedPath ?? ""] ?? current?.review ?? []
    }
    var recipeSignature:String {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        let normal=String(data:(try? encoder.encode(normalRecipe)) ?? Data(),encoding:.utf8) ?? ""
        let marked=String(data:(try? encoder.encode(photoRecipes(checked.map(\.path)))) ?? Data(),encoding:.utf8) ?? ""
        let learned=String(data:(try? encoder.encode(adjustmentsFor(checked.map(\.path)))) ?? Data(),encoding:.utf8) ?? ""
        return normal+marked+learned+checked.map(\.path).sorted().joined(separator:"|")
    }
    var currentExport:String? {
        if let variants=current?.variants { return variants[cropMode]?.jpeg }
        return cropMode == "centered" ? current?.jpeg : nil
    }
    var matchesExport:Bool {
        guard let photo=current,let path=currentExport else { return false }
        return exportSignatures[photo.source]==recipeSignature && FileManager.default.fileExists(atPath:path)
    }
    var currentEdited:String? {
        if matchesExport { return currentExport }
        return previewSource==selectedPath ? (previewPath ?? currentExport) : currentExport
    }
    var currentOriginal:String? {
        if previewSource==selectedPath,let p=previewOriginal { return p }
        if let p=current?.before { return p }
        guard inputs.indices.contains(selected) else { return nil }
        let p=root.appendingPathComponent("work/previews/\(inputs[selected].deletingPathExtension().lastPathComponent).jpg").path
        return FileManager.default.fileExists(atPath:p) ? p : nil
    }
    var frameKnown:Bool { !checked.isEmpty && checked.allSatisfy { knownFrames[$0.path] != nil } }
    var recipe:[String:Any] { ["exposure":exposure,"contrast":contrast,"whites":whites,"warmth":warmth,"fill":fill,"sharpness":sharpness,"denoise":denoise,"shadow":shadow,"aspect":aspect,"size":size,"framing":cropMode,"shadow_method":enhancedShadow ? "local":"classic","shadow_detection":softShadowDetection ? "soft":"dark","package_cleanup":editingPackagePath != nil,"centered_scale":independentScale ? "independent":"shared","small_parts":smallParts,"recover_parts":recoverParts,"recover_white":recoverWhite,"recover_dark":recoverDark] }
    init() {
        root=Bundle.main.bundleURL.deletingLastPathComponent()
        outputParent=root.appendingPathComponent("output/deliverables")
        output=outputParent.appendingPathComponent("catalog")
        metadata=root.appendingPathComponent("work/shoots/catalog")
        BrickStudioDelegate.cleanups[previewSessionID] = { [weak self] in self?.shutdown() }
        loadQueue()
        loadPackaging()
        loadAdjustments()
        loadFolderRecipes()
        for job in queue.jobs where folderRecipes[job.folder]==nil {
            folderRecipes[job.folder]=job.started_recipe ?? job.override ?? queue.shared_recipe
        }
        // After packaging loads, so saving the queue keeps packaging recipes.
        if !queueRunning { settleFinishedBatches() }
        // Photos start empty; queue metadata is available.
    }
    func resetDefaults() {
        exposure=0.65;contrast=0;whites=0;warmth=0.25;fill=0.84
        sharpness=0.75;denoise=0;shadow=2;enhancedShadow=false;softShadowDetection=false;packageCleanup=false;smallParts=false;recoverParts=false;recoverWhite=false;recoverDark=false
        aspect="square";size=2400;maskMode="efficient";maskBackend="mps";exportFraming="centered";previewFraming="centered";independentScale=false
    }

    func refresh() {
        guard let data=try? Data(contentsOf:metadata.appendingPathComponent("manifest.json")),let manifest=try? JSONDecoder().decode(Manifest.self,from:data) else { return }
        var bySource=Dictionary(uniqueKeysWithValues:photos.map { ($0.source,$0) })
        for photo in manifest.records {
            bySource[photo.source]=photo
            if let b=photo.bbox,let sz=photo.source_size { knownFrames[photo.source]=["source":photo.source,"bbox":b,"source_size":sz] }
            if running { exportSignatures[photo.source]=activeExportSignature }
        }
        photos=inputs.compactMap { bySource[$0.path] }
        if running { batchCompleted=manifest.records.count }
    }
    func toggle(_ url:URL) {
        if included.contains(url.path) { included.remove(url.path) } else { included.insert(url.path) }
    }
    var photoAnchor:Int?
    /// Plain click views a photo. Shift-click checks the range from the last clicked photo;
    /// Command-click checks or unchecks one. Edits apply to the checked photos.
    func clickPhoto(_ index:Int,shift:Bool,command:Bool) {
        guard inputs.indices.contains(index) else { return }
        if !running {
            if shift,let anchor=photoAnchor,inputs.indices.contains(anchor) {
                let range=Set(inputs[min(anchor,index)...max(anchor,index)].map(\.path))
                included=command ? included.union(range):range
            } else if command {
                toggle(inputs[index]);photoAnchor=index
            } else { photoAnchor=index }
        }
        selected=index
    }
    func selectAll() { included=Set(inputs.map(\.path)) }
    func deselectAll() { included=[] }
    func importPhotos() {
        let panel=NSOpenPanel()
        panel.canChooseDirectories=true;panel.canChooseFiles=true;panel.allowsMultipleSelection=true;panel.prompt="Import shoot"
        guard panel.runModal() == .OK else { return }
        var found:[URL]=[]
        for url in panel.urls {
            var directory:ObjCBool=false
            FileManager.default.fileExists(atPath:url.path,isDirectory:&directory)
            let candidates=directory.boolValue ? ((try? FileManager.default.contentsOfDirectory(at:url,includingPropertiesForKeys:nil)) ?? []) : [url]
            found+=candidates.filter { extensions.contains($0.pathExtension.lowercased()) }
        }
        leavePackagingEditor();stopPreviewWorker();previewFolderID=nil
        queueVisible=false;recipeTarget=nil;loadPhotos(found)
        if autoLearn { requestLearning(inputs.map(\.path)) }
    }
    func loadPhotos(_ found:[URL]) {
        guard !found.isEmpty else { error="No supported RAW or image files were found.";return }
        clearStaleAdjustments(found.map(\.path))
        inputs=Array(Set(found.map { $0.resolvingSymlinksInPath() })).sorted { $0.lastPathComponent<$1.lastPathComponent }
        photos=[];knownFrames=[:];exportSignatures=[:];included=Set(inputs.map(\.path));selected=0
        previewPath=nil;previewOriginal=nil;previewSource="";previewReviews=[:]
        progress=0;batchCompleted=0;status="\(inputs.count) photos imported · all selected"
        activatePhotoRecipe()
        if previewProcess==nil { startPreviewWorker() } else { schedulePreview() }
    }
    func chooseDestination() {
        let panel=NSOpenPanel();panel.canChooseFiles=false;panel.canChooseDirectories=true;panel.canCreateDirectories=true;panel.prompt="Use folder"
        if panel.runModal() == .OK,let url=panel.url { outputParent=url;status="Next export: \(url.lastPathComponent)" }
    }
    func startPreviewWorker() {
        guard previewProcess==nil else { return }
        do {
            try FileManager.default.createDirectory(at:root.appendingPathComponent("work"),withIntermediateDirectories:true)
            let task=Process(),input=Pipe(),outputPipe=Pipe()
            previewGeneration+=1
            let generation=previewGeneration
            task.executableURL=root.appendingPathComponent(".venv/bin/python")
            task.arguments=[root.appendingPathComponent("preview_worker.py").path,"--session-dir",previewDirectory.path,"--owner-pid",String(ProcessInfo.processInfo.processIdentifier)]
            task.currentDirectoryURL=root
            task.standardInput=input;task.standardOutput=outputPipe
            let errorURL=root.appendingPathComponent("work/preview-worker.log")
            FileManager.default.createFile(atPath:errorURL.path,contents:nil)
            task.standardError=try FileHandle(forWritingTo:errorURL)
            previewInput=input.fileHandleForWriting;previewOutput=outputPipe.fileHandleForReading
            outputPipe.fileHandleForReading.readabilityHandler = { [self] handle in
                let data=handle.availableData
                if !data.isEmpty { Task { @MainActor in self.consumePreview(data,generation:generation) } }
            }
            task.terminationHandler = { [self] ended in
                Task { @MainActor in
                    if generation==self.previewGeneration && self.previewEnabled {
                        self.previewLoading=false
                        if let active=self.batchPreviewActive {
                            self.batchPreviewFailures[active.id]=active.signature
                            self.batchPreviewNotes[active.id]="Preview stopped · retry when ready"
                            self.batchPreviewActive=nil
                        }
                        self.previewProcess=nil;self.previewEnabled=false
                        self.previewNote="Preview renderer stopped. Reopen the app to retry."
                    }
                }
            }
            previewProcess=task;previewEnabled=true
            try task.run();schedulePreview()
        } catch { previewProcess=nil;previewEnabled=false;previewNote=error.localizedDescription }
    }
    func stopPreviewWorker() {
        previewEnabled=false;previewGeneration+=1;previewSerial+=1
        batchPreviewActive=nil;batchPreviewDetailSerial=nil
        previewWork?.cancel();previewOutput?.readabilityHandler=nil
        try? previewInput?.close();try? previewOutput?.close()
        if previewProcess?.isRunning==true { previewProcess?.terminate() }
        previewInput=nil;previewOutput=nil;previewProcess=nil;previewBuffer=Data();previewLoading=false
    }
    func schedulePreview() {
        if applyingRecipe || switchingPhotoRecipe { return }
        if queueVisible {
            if !queueRunning { saveRecipeTarget() }
            scheduleBatchPreviews()
            return
        }
        saveCurrentAdjustment()
        guard previewEnabled,!running,!queueRunning,selectedPath != nil else { return }
        previewSerial+=1
        let serial=previewSerial
        previewWork?.cancel()
        if matchesExport {
            previewLoading=false;previewNote="Export · \(max(current?.width ?? size,current?.height ?? size)) px"
            return
        }
        previewLoading=true;previewNote="Updating full-detail preview…"
        let work=DispatchWorkItem { [self] in self.sendPreview(serial) }
        previewWork=work
        DispatchQueue.main.asyncAfter(deadline:.now()+0.18,execute:work)
    }
    func sendPreview(_ serial:Int) {
        guard serial==previewSerial,previewEnabled,!running,let source=selectedPath else { return }
        var request:[String:Any]=["id":serial,"source":source,"settings":recipe,"mask_mode":maskMode,"mask_backend":maskBackend]
        request["batch_geometry"]=checked.filter { editingPackagePath != nil ? $0.path==source:packagingRecipeFor($0.path)==nil }.compactMap { knownFrames[$0.path] }
        do {
            var data=try JSONSerialization.data(withJSONObject:request);data.append(10)
            try previewInput?.write(contentsOf:data)
        } catch { previewLoading=false;previewNote=error.localizedDescription }
    }
    func consumePreview(_ data:Data,generation:Int) {
        guard generation==previewGeneration else { return }
        previewBuffer.append(data)
        while let newline=previewBuffer.firstIndex(of:10) {
            let line=Data(previewBuffer[..<newline]);previewBuffer.removeSubrange(...newline)
            guard let event=(try? JSONSerialization.jsonObject(with:line)) as? [String:Any] else { continue }
            guard let id=event["id"] as? Int else { continue }
            if let active=batchPreviewActive,id==active.serial {
                if active.isProbe { consumeGeometryProbe(event,active:active) } else { consumeBatchPreview(event,active:active) }
                continue
            }
            guard id==previewSerial else { continue }
            let kind=event["event"] as? String
            if kind=="preview_progress" { previewNote=(event["phase"] as? String ?? "Preparing preview")+"…" }
            if kind=="preview_source",let source=event["source"] as? String {
                if previewSource != source { previewPath=nil }
                previewSource=source;previewOriginal=event["path"] as? String
            }
            if kind=="preview_result",let source=event["source"] as? String,source==selectedPath {
                previewSource=source;previewPath=event["path"] as? String;previewOriginal=event["original"] as? String
                previewSeconds=event["seconds"] as? Double ?? 0;previewLoading=false
                previewReviews[source]=event["review"] as? [String] ?? []
                if let box=event["bbox"] as? [Int],let sz=event["source_size"] as? [Int] { knownFrames[source]=["source":source,"bbox":box,"source_size":sz] }
                previewNote=(frameKnown ? "Live preview · " : "Provisional framing · ")+"\(max(event["width"] as? Int ?? size,event["height"] as? Int ?? size)) px"
            }
            if kind=="preview_error" { previewLoading=false;previewNote=event["message"] as? String ?? "Preview failed" }
        }
    }
    func run() {
        let batch=checked
        guard !running,!learningBusy,!batch.isEmpty else { return }
        activeExportSignature=recipeSignature
        stopPreviewWorker()
        do {
            var replacing=false
            if let id=previewFolderID,
               let job=queue.jobs.first(where:{$0.id==id}),
               let outPath=job.output,!outPath.isEmpty,
               let metaPath=job.metadata,!metaPath.isEmpty,
               FileManager.default.fileExists(atPath:outPath),
               FileManager.default.fileExists(atPath:(metaPath as NSString).appendingPathComponent("manifest.json")) {
                // Preview-all → Process selected: write back into that batch's export.
                output=URL(fileURLWithPath:outPath)
                metadata=URL(fileURLWithPath:metaPath)
                replacing=true
            } else {
                let stamp=ISO8601DateFormatter().string(from:Date()).replacingOccurrences(of:":",with:"-")
                output=outputParent.appendingPathComponent("Shoot-\(stamp)-\(UUID().uuidString.prefix(4))")
                try FileManager.default.createDirectory(at:output,withIntermediateDirectories:true)
                metadata=root.appendingPathComponent("work/shoots/\(output.lastPathComponent)")
                try FileManager.default.createDirectory(at:metadata,withIntermediateDirectories:true)
            }
            let list=metadata.appendingPathComponent("inputs.json")
            try JSONEncoder().encode(batch.map(\.path)).write(to:list)
            let log=metadata.appendingPathComponent("processing.log")
            if !FileManager.default.fileExists(atPath:log.path) { FileManager.default.createFile(atPath:log.path,contents:nil) }
            let handle=try FileHandle(forWritingTo:log)
            try handle.seekToEnd()
            logRead=try FileHandle(forReadingFrom:log);logBuffer=Data()
            try logRead?.seekToEnd()
            let task=Process();task.executableURL=root.appendingPathComponent(".venv/bin/python");task.currentDirectoryURL=root
            let photoSettings=metadata.appendingPathComponent("photo-recipes.json")
            try JSONEncoder().encode(photoRecipes(batch.map(\.path))).write(to:photoSettings)
            let r=normalRecipe
            var args:[String]=[root.appendingPathComponent("studio.py").path,"--input-list",list.path,"--output",output.path,"--metadata",metadata.path]
            args += ["--framing",r.framing,"--size",String(r.size),"--aspect",r.aspect,
                      "--exposure",String(r.exposure),"--warmth",String(r.warmth),"--contrast",String(r.contrast),
                      "--whites",String(r.whites ?? 0),"--shadow",String(r.shadow),"--shadow-method",r.shadow_method,
                      "--fill",String(r.fill),"--sharpness",String(r.sharpness),"--denoise",String(r.denoise),
                      "--mask-mode",r.mask_mode,"--mask-backend",r.mask_backend ?? "mps",
                      "--centered-scale",r.centered_scale ?? "shared","--shadow-detection",r.shadow_detection ?? "dark","--photo-recipes",photoSettings.path]
            let adjustmentsFile=metadata.appendingPathComponent("adjustment-recipes.json")
            try JSONEncoder().encode(adjustmentsFor(batch.map(\.path))).write(to:adjustmentsFile,options:.atomic)
            args += ["--adjustment-recipes",adjustmentsFile.path]
            if r.small_parts == true { args.append("--small-parts") }
            if r.recover_parts == true { args.append("--recover-parts") }
            if r.recover_white == true { args.append("--recover-white") }
            if r.recover_dark == true { args.append("--recover-dark") }
            if replacing { args.append("--replace") }
            task.arguments=args
            task.standardOutput=handle;task.standardError=handle
            task.terminationHandler = { [self] ended in
                Task { @MainActor in
                    self.timer?.invalidate();self.poll();self.refresh();self.running=false
                    self.timer=nil;self.process=nil;try? handle.close();try? self.logRead?.close();self.logRead=nil
                    if ended.terminationStatus==0 {
                        self.progress=1;self.photoProgress=1;self.eta=0
                        self.status=replacing
                            ? "\(self.batchCompleted) photos updated in the existing export"
                            : "\(self.batchCompleted) selected photos exported · JPEG + PNG"
                        if replacing,let id=self.previewFolderID {
                            self.batchPreviewSignatures[id]=nil
                            self.batchImages[id]=nil
                        }
                    }
                    else if ended.terminationReason == .uncaughtSignal { self.status="Stopped · completed photos are saved" }
                    else { self.status="Finished with errors · \(self.batchCompleted) photos saved";self.error="Some photos could not be processed. Details are saved in the app’s work/shoots folder." }
                    self.startPreviewWorker()
                }
            }
            running=true;progress=0;photoProgress=0;batchCompleted=0;batchTotal=batch.count;activePhoto=1
            batchStarted=Date();photoStarted=Date();batchElapsed=0;photoElapsed=0;finishing=false
            activeName=batch[0].lastPathComponent;phase="Preparing";eta=Double(batch.count)*(maskBackend=="mps" ? 5:28)
            status=replacing ? "Updating \(batch.count) selected photos in the existing export" : "Processing \(batch.count) selected photos"
            process=task;try task.run()
            timer=Timer.scheduledTimer(withTimeInterval:0.2,repeats:true) { [self] _ in Task { @MainActor in self.poll() } }
        } catch { running=false;process=nil;self.error=error.localizedDescription;startPreviewWorker() }
    }
    func poll() {
        if let data=try? logRead?.readToEnd() {
            logBuffer.append(data)
            while let newline=logBuffer.firstIndex(of:10) {
                let line=Data(logBuffer[..<newline]);logBuffer.removeSubrange(...newline)
                guard let event=(try? JSONSerialization.jsonObject(with:line)) as? [String:Any] else { continue }
                if event["stage"] as? String == "progress" {
                    let number=event["photo_number"] as? Int ?? activePhoto
                    photoStarted=Date().addingTimeInterval(-(event["photo_elapsed"] as? Double ?? 0))
                    activePhoto=number;activeName=event["name"] as? String ?? activeName
                    phase=event["phase"] as? String ?? phase
                    finishing=phase=="Aligning framing and packaging" || phase=="Complete"
                    photoElapsed=event["photo_elapsed"] as? Double ?? photoElapsed
                    photoProgress=event["photo_progress"] as? Double ?? photoProgress
                    progress=event["batch_progress"] as? Double ?? progress
                    etaValue=event["eta_seconds"] as? Double ?? eta;etaReceived=Date()
                    batchCompleted=event["completed"] as? Int ?? batchCompleted
                }
                if event["stage"] as? String == "photo_done" { refresh() }
            }
        }
        if running {
            if let start=batchStarted { batchElapsed=Date().timeIntervalSince(start) }
            if !finishing,let start=photoStarted { photoElapsed=Date().timeIntervalSince(start) }
            if let received=etaReceived { eta=max(0,etaValue-Date().timeIntervalSince(received)) }
        }
    }
    func openOutput() {
        if let id=previewFolderID ?? (queueVisible ? selectedBatchID:nil),queue.jobs.first(where:{$0.id==id})?.output != nil { openExport(id);return }
        let target=queueVisible ? URL(fileURLWithPath:queue.output_root):output
        guard FileManager.default.fileExists(atPath:target.path) else {
            error="The export folder is not available:\n\(target.path)\n\nIf it is on an external drive, connect the drive first.";return
        }
        NSWorkspace.shared.open(target)
    }
    func cancel() { process?.terminate() }
    func shutdown() {
        learningPending=[]
        learningTimer?.invalidate();learningTimer=nil
        if learningProcess?.isRunning==true { learningProcess?.terminate() }
        if queueRunning { pauseQueue() }
        queueTimer?.invalidate();queueTimer=nil
        let renderer=previewProcess
        stopPreviewWorker()
        // Wait for its file writes to stop before deleting this session only.
        if renderer?.isRunning==true { renderer?.waitUntilExit() }
        try? FileManager.default.removeItem(at:previewDirectory)
        BrickStudioDelegate.cleanups.removeValue(forKey:previewSessionID)
    }

    var batchRecipe:BatchRecipe {
        BatchRecipe(exposure:exposure,contrast:contrast,warmth:warmth,fill:fill,sharpness:sharpness,denoise:denoise,shadow:shadow,
                    aspect:aspect,framing:exportFraming,mask_mode:maskMode,shadow_method:enhancedShadow ? "local":"classic",size:size,package_cleanup:editingPackagePath != nil ? true:nil,centered_scale:independentScale ? "independent":nil,whites:whites==0 ? nil:whites,small_parts:smallParts ? true:nil,mask_backend:maskBackend=="mps" ? nil:maskBackend,shadow_detection:softShadowDetection ? "soft":nil,recover_parts:recoverParts ? true:nil,recover_white:recoverWhite ? true:nil,recover_dark:recoverDark ? true:nil)
    }
    func applyRecipe(_ r:BatchRecipe) {
        applyingRecipe=true
        exposure=r.exposure;contrast=r.contrast;whites=r.whites ?? 0;warmth=r.warmth;fill=r.fill;sharpness=r.sharpness;denoise=r.denoise;shadow=r.shadow
        aspect=r.aspect;exportFraming=r.framing;maskMode=r.mask_mode;maskBackend=r.mask_backend ?? "mps";enhancedShadow=r.shadow_method=="local";softShadowDetection=r.shadow_detection=="soft";size=r.size;packageCleanup=editingPackagePath != nil;independentScale=r.centered_scale=="independent";smallParts=r.small_parts ?? false
        recoverParts=r.recover_parts ?? false
        recoverWhite=r.recover_white ?? false
        recoverDark=r.recover_dark ?? false
        appliedRecipe=batchRecipe;applyingRecipe=false;schedulePreview()
    }
    func saveQueue() {
        for i in queue.jobs.indices where queue.jobs[i].started_recipe==nil && queue.jobs[i].archived != true {
            queue.jobs[i].photo_recipes=photoRecipes(queue.jobs[i].sources,freeze:false)
            queue.jobs[i].adjustment_recipes=adjustmentsFor(queue.jobs[i].sources)
        }
        do {
            try FileManager.default.createDirectory(at:queueURL.deletingLastPathComponent(),withIntermediateDirectories:true)
            try JSONEncoder().encode(queue).write(to:queueURL,options:.atomic)
        } catch { self.error="Could not save the batch queue: "+error.localizedDescription }
    }
    func loadQueue() {
        if let data=try? Data(contentsOf:queueURL),let saved=try? JSONDecoder().decode(BatchQueueState.self,from:data) {
            queue=saved
            if queueRunning {
                let alive=queue.worker_pid.map { Darwin.kill(pid_t($0),0)==0 } ?? false
                if alive { watchQueue() }
                else { queue.status="paused";queue.phase="Resume your saved queue";saveQueue() }
            }
        }
        if queue.output_root.isEmpty { queue.output_root=outputParent.path }
    }
    func openQueue() {
        guard !running else { return }
        leavePackagingEditor();stopPreviewWorker()
        if queue.jobs.isEmpty { queue.shared_recipe=BatchRecipe.newFolder(normalRecipe) }
        queueVisible=true
        refreshBatchFolders(onlyUnstarted:true)
        if let id=previewFolderID,activeBatches.contains(where:{$0.id==id}) { selectedBatchID=id }
        if selectedBatch==nil { selectedBatchID=activeBatches.first?.id }
        if let id=selectedBatchID { selectBatch(id) }
        else { recipeTarget=nil;applyRecipe(queue.shared_recipe) }
        if !queueRunning { startPreviewWorker() }
    }
    func selectBatch(_ id:String) {
        guard let job=activeBatches.first(where:{$0.id==id}) else { return }
        if selectedBatchID != id {
            batchDetailImage=nil;batchDetailSignature="";batchDetailJobID=nil
        }
        selectedBatchID=id
        recipeTarget=id
        applyRecipe(batchPreviewRecipe(job))
        loadCompletedBatchImages();scheduleBatchPreviews()
    }
    func angleIndex(_ job:BatchJob)->Int { min(max(0,batchAngles[job.id] ?? 0),max(0,job.sources.count-1)) }
    func previewSource(_ job:BatchJob)->String? { job.sources.isEmpty ? nil:job.sources[angleIndex(job)] }
    /// Turntable angle from a name such as 11012-1_010deg_….dng.
    nonisolated static func degrees(_ source:String)->Int? {
        let name=URL(fileURLWithPath:source).lastPathComponent
        guard let range=name.range(of:#"_(\d{1,3})deg"#,options:.regularExpression) else { return nil }
        return Int(name[range].filter(\.isNumber))
    }
    func angleLabel(_ job:BatchJob,_ index:Int)->String {
        guard job.sources.indices.contains(index) else { return "" }
        return Studio.degrees(job.sources[index]).map { String(format:"%03d°",$0) } ?? "Photo \(index+1)"
    }
    func setAngle(_ id:String,_ index:Int) {
        guard !queueRunning,!running,let job=activeBatches.first(where:{$0.id==id}),!job.sources.isEmpty else { return }
        let clamped=min(max(0,index),job.sources.count-1)
        guard clamped != angleIndex(job) else { return }
        batchAngles[id]=clamped
        // The sliders follow the photo being viewed; edits still apply to the whole folder.
        if selectedBatchID==id { applyRecipe(batchPreviewRecipe(job)) }
        scheduleBatchPreviews()
    }
    func stepAngle(_ delta:Int) {
        guard let job=selectedBatch,job.sources.count>1 else { return }
        let n=job.sources.count
        setAngle(job.id,((angleIndex(job)+delta)%n+n)%n)
    }
    func jumpAngle(_ target:Int) {
        guard let job=selectedBatch else { return }
        let scored=job.sources.enumerated().compactMap { i,source in Studio.degrees(source).map { d in (i,min(abs(d-target),360-abs(d-target))) } }
        if let best=scored.min(by:{$0.1<$1.1}) { setAngle(job.id,best.0) }
    }
    func flashLearningNote() {
        learningNoteFresh=true
        let stamp=UUID();learningFlashStamp=stamp
        DispatchQueue.main.asyncAfter(deadline:.now()+7) { if self.learningFlashStamp==stamp { self.learningNoteFresh=false } }
    }
    func editSharedRecipe() {
        if let job=activeBatches.first(where:{$0.override==nil && $0.started_recipe==nil}) {
            if selectedBatchID != job.id { batchDetailImage=nil;batchDetailSignature="";batchDetailJobID=nil }
            selectedBatchID=job.id
        }
        recipeTarget=nil;applyRecipe(queue.shared_recipe)
    }
    func editOverride(_ id:String) {
        guard !queueRunning,let job=queue.jobs.first(where:{$0.id==id}),job.started_recipe==nil else { return }
        selectBatch(id)
    }
    func showSingleShoot() {
        stopPreviewWorker();queueVisible=false
        if let id=previewFolderID,let job=queue.jobs.first(where:{$0.id==id}) {
            recipeTarget=id;applyRecipe(job.started_recipe ?? job.override ?? queue.shared_recipe)
        } else { recipeTarget=nil;previewFolderID=nil }
        activatePhotoRecipe()
        if !inputs.isEmpty { startPreviewWorker() }
    }
    func saveRecipeTarget() {
        guard editingPackagePath == nil,editingAdjustmentPath == nil else { return }
        guard let previous=appliedRecipe,batchRecipe != previous else { return }
        let values=toneValues(batchRecipe).filter { toneValues(previous)[$0.key] != $0.value }
        let options=photoOptions(batchRecipe).filter { photoOptions(previous)[$0.key] != $0.value }
        if let id=recipeTarget,queue.jobs.contains(where:{$0.id==id}) {
            for target in actionTargets(id) {
                guard let i=queue.jobs.firstIndex(where:{$0.id==target}),queue.jobs[i].started_recipe==nil else { continue }
                let base=queue.jobs[i].override ?? queue.shared_recipe
                queue.jobs[i].override=mergedOptions(mergedTone(base,values),options)
                recordPhotoEdits(queue.jobs[i].sources,values:values,options:options,inherit:true)
            }
        } else if recipeTarget==nil {
            queue.shared_recipe=mergedOptions(mergedTone(queue.shared_recipe,values),options)
            recordPhotoEdits(queue.jobs.filter{$0.override==nil && $0.started_recipe==nil}.flatMap(\.sources),values:values,options:options,inherit:true)
        }
        appliedRecipe=batchRecipe
        storeAdjustments();storePackaging();rememberFolderRecipes();saveQueue()
    }
    func useShared(_ id:String) {
        for target in actionTargets(id) {
            guard let i=queue.jobs.firstIndex(where:{$0.id==target}),queue.jobs[i].started_recipe==nil else { continue }
            clearAdjustments(queue.jobs[i].sources)
            queue.jobs[i].override=nil
        }
        recipeTarget=nil;applyRecipe(queue.shared_recipe);saveQueue()
    }
    func chooseBatches() {
        let panel=NSOpenPanel();panel.canChooseFiles=false;panel.canChooseDirectories=true;panel.allowsMultipleSelection=true;panel.prompt="Add batches"
        if panel.runModal() == .OK { addBatchFolders(panel.urls) }
    }
    func batchFiles(_ folder:URL)->[URL] {
        ((try? FileManager.default.contentsOfDirectory(at:folder,includingPropertiesForKeys:[.isRegularFileKey],options:.skipsHiddenFiles)) ?? [])
            .filter { extensions.contains($0.pathExtension.lowercased()) && ((try? $0.resourceValues(forKeys:[.isRegularFileKey]).isRegularFile)==true) }
            .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
    }
    /// Rescans batch folders for photos added or removed on disk, keeping each folder's recipe.
    /// onlyUnstarted refreshes untouched queued batches silently; otherwise finished, paused
    /// or stopped batches are queued to export afresh into a new folder, leaving earlier exports as they are.
    func refreshBatchFolders(_ ids:[String]?=nil,onlyUnstarted:Bool=false) {
        guard !queueRunning,!running else { return }
        var changes:[String]=[],unavailable:[String]=[]
        for i in queue.jobs.indices where ids?.contains(queue.jobs[i].id) ?? true {
            let job=queue.jobs[i]
            if job.archived == true { continue }
            let started=job.started_recipe != nil || job.status != "queued"
            if onlyUnstarted && started { continue }
            var isDir:ObjCBool=false
            let sources=FileManager.default.fileExists(atPath:job.folder,isDirectory:&isDir) && isDir.boolValue ? batchFiles(URL(fileURLWithPath:job.folder)).map(\.path):[]
            guard !sources.isEmpty else { unavailable.append(job.name);continue }
            guard sources != job.sources else { continue }
            queue.jobs[i].sources=sources
            if started {
                // A different photo set needs a new framing plan, so never resume or overwrite the old export.
                let run=UUID().uuidString.prefix(8)
                queue.jobs[i].override=job.started_recipe ?? job.override;queue.jobs[i].started_recipe=nil
                queue.jobs[i].status="queued";queue.jobs[i].message=nil
                queue.jobs[i].completed=0;queue.jobs[i].errors=0;queue.jobs[i].attempts=0
                queue.jobs[i].output=URL(fileURLWithPath:queue.output_root).appendingPathComponent("\(job.name)-\(run)").path
                queue.jobs[i].metadata=root.appendingPathComponent("work/shoots/queue-\(job.id)-\(run)").path
            }
            changes.append("\(job.name): \(job.sources.count) → \(sources.count) photos"+(started ? " · will export again into a new folder":""))
        }
        if !changes.isEmpty {
            saveQueue()
            if let id=selectedBatchID,queue.jobs.contains(where:{$0.id==id}) { selectBatch(id) }
            scheduleBatchPreviews()
        }
        guard !onlyUnstarted else { return }
        var lines=changes.isEmpty ? ["No new or removed photos found."]:changes
        if !unavailable.isEmpty { lines.append("Not found or empty, left unchanged: "+unavailable.joined(separator:", ")) }
        error=lines.joined(separator:"\n")
    }
    func addBatchFolders(_ urls:[URL]) {
        guard !queueRunning,!running else { return }
        openQueue()
        var added:[BatchJob]=[]
        func files(_ folder:URL)->[URL] { batchFiles(folder) }
        for dropped in urls {
            let folder=dropped.resolvingSymlinksInPath()
            var isDir:ObjCBool=false
            guard FileManager.default.fileExists(atPath:folder.path,isDirectory:&isDir),isDir.boolValue else { continue }
            let folders=files(folder).isEmpty ? ((try? FileManager.default.contentsOfDirectory(at:folder,includingPropertiesForKeys:[.isDirectoryKey],options:.skipsHiddenFiles)) ?? []).filter { (try? $0.resourceValues(forKeys:[.isDirectoryKey]).isDirectory)==true }.sorted{$0.lastPathComponent<$1.lastPathComponent} : [folder]
            for item in folders {
                let sources=files(item).map(\.path)
                guard !sources.isEmpty,!queue.jobs.contains(where:{$0.folder==item.path && !["complete","error","stopped"].contains($0.status)}),!added.contains(where:{$0.folder==item.path}) else { continue }
                let recipe=folderRecipes[item.path] ?? BatchRecipe.newFolder(queue.shared_recipe)
                added.append(BatchJob(name:item.lastPathComponent,folder:item.path,sources:sources,override:recipe==queue.shared_recipe ? nil:recipe))
            }
        }
        // New imports join the end of the working row, keeping their own order,
        // and stay ahead of the folders marked completed.
        let tail=queue.jobs.firstIndex(where:{$0.archived==true}) ?? queue.jobs.count
        queue.jobs.insert(contentsOf:added,at:tail)
        if autoLearn { requestLearning(added.flatMap(\.sources)) }
        if added.isEmpty { error="No new product folders with supported photos were found." }
        queue.status="idle";queue.progress=nil;rememberFolderRecipes();saveQueue()
        if let first=added.first { selectBatch(first.id) }
        else if selectedBatch==nil,let first=queue.jobs.first { selectBatch(first.id) }
        scheduleBatchPreviews()
    }
    func receiveFolders(_ providers:[NSItemProvider])->Bool {
        guard !running,!queueRunning else { return false }
        for provider in providers {
            provider.loadItem(forTypeIdentifier:UTType.fileURL.identifier,options:nil) { item,_ in
                let url:URL?
                if let data=item as? Data { url=URL(dataRepresentation:data,relativeTo:nil) }
                else { url=item as? URL }
                if let url=url { Task { @MainActor in self.addBatchFolders([url]) } }
            }
        }
        return true
    }
    func removeBatch(_ id:String) { removeBatches([id]) }
    func removeBatches(_ ids:[String]) {
        guard !queueRunning else { return }
        let ids=Set(ids)
        for job in queue.jobs where ids.contains(job.id) { folderRecipes[job.folder]=job.started_recipe ?? job.override ?? queue.shared_recipe }
        rememberFolderRecipes()
        queue.jobs.removeAll{ids.contains($0.id)}
        for id in ids { batchImages[id]=nil;batchPreviewSignatures[id]=nil }
        if let id=previewFolderID,ids.contains(id) { previewFolderID=nil }
        if let id=selectedBatchID,ids.contains(id) {
            batchDetailImage=nil;batchDetailSignature="";batchDetailJobID=nil
            selectedBatchID=nil
            if let next=activeBatches.first { selectBatch(next.id) } else { editSharedRecipe() }
        }
        if let id=recipeTarget,ids.contains(id) { editSharedRecipe() }
        batchSelectionAnchor=nil;draggedBatchIDs=[];saveQueue();scheduleBatchPreviews()
    }
    func previewBatch(_ requested:BatchJob) {
        guard !queueRunning,!running,let job=queue.jobs.first(where:{$0.id==requested.id}) else { return }
        leavePackagingEditor();stopPreviewWorker()
        queueVisible=false;previewFolderID=job.id;recipeTarget=job.id;selectedBatchID=job.id
        let geometry=knownFrames
        applyingRecipe=true
        loadPhotos(job.sources.map{URL(fileURLWithPath:$0)})
        knownFrames.merge(geometry) { current,_ in current }
        if let viewed=previewSource(job),let i=inputs.firstIndex(where:{$0.path==URL(fileURLWithPath:viewed).resolvingSymlinksInPath().path}) { selected=i }
        applyingRecipe=false
        mode="Edited"
        applyRecipe(job.started_recipe ?? job.override ?? queue.shared_recipe)
        activatePhotoRecipe();schedulePreview()
    }
    func loadPackaging() {
        guard FileManager.default.fileExists(atPath:packagingURL.path) else { return }
        do { packaging=try JSONDecoder().decode(PackagingLibrary.self,from:Data(contentsOf:packagingURL)) }
        catch { packagingAvailable=false;self.error="Could not read packaging settings. The saved file has been kept." }
    }
    func storePackaging() {
        guard packagingAvailable else { return }
        do {
            try FileManager.default.createDirectory(at:packagingURL.deletingLastPathComponent(),withIntermediateDirectories:true)
            try JSONEncoder().encode(packaging).write(to:packagingURL,options:.atomic)
        } catch { self.error="Could not save packaging settings: "+error.localizedDescription }
    }
    func packagingRecipeFor(_ path:String,job:BatchJob?=nil)->BatchRecipe? {
        let context=job ?? queue.jobs.first { $0.id==previewFolderID && $0.sources.contains(path) }
        if let context=context,context.started_recipe != nil { return context.photo_recipes?[path] }
        guard packaging.marked.contains(path) else { return nil }
        var r=packaging.overrides[path] ?? packaging.recipe;r.package_cleanup=true;return r
    }
    func photoRecipes(_ sources:[String],freeze:Bool=true)->[String:BatchRecipe] {
        var result:[String:BatchRecipe]=[:]
        for path in sources {
            if freeze,let r=packagingRecipeFor(path) { result[path]=r }
            else if !freeze,packaging.marked.contains(path) {
                var r=packaging.overrides[path] ?? packaging.recipe;r.package_cleanup=true;result[path]=r
            }
        }
        return result
    }
    func leavePackagingEditor() {
        guard editingPackagePath != nil || editingAdjustmentPath != nil else { return }
        switchingPhotoRecipe=true;editingPackagePath=nil;editingAdjustmentPath=nil;applyRecipe(regularRecipe);switchingPhotoRecipe=false
    }
    func activatePhotoRecipe() {
        guard !applyingRecipe,!switchingPhotoRecipe,!queueVisible else { return }
        let next=selectedPath.flatMap { packagingRecipeFor($0) == nil ? nil:$0 }
        let learned=selectedPath
        guard next != editingPackagePath || learned != editingAdjustmentPath else { return }
        leavePackagingEditor()
        if let path=selectedPath, next != nil || learned != nil {
            switchingPhotoRecipe=true;regularRecipe=batchRecipe;editingPackagePath=next;editingAdjustmentPath=learned
            applyRecipe(adjustedRecipe(packagingRecipeFor(path) ?? regularRecipe,path:path));switchingPhotoRecipe=false
        }
    }
    func markPackaging(_ path:String,_ marked:Bool) {
        guard !running,!queueRunning,!recipeLocked,packagingAvailable else { return }
        if marked { packaging.marked.insert(path) } else { packaging.marked.remove(path) }
        leavePackagingEditor();clearAdjustments([path])
        storePackaging();saveQueue();activatePhotoRecipe();schedulePreview()
        if autoLearn { requestLearning([path]) }
    }
    func saveCurrentPackagingRecipe() {
        guard let path=editingPackagePath,!running,!queueRunning,!recipeLocked else { return }
        var r=batchRecipe;r.package_cleanup=true
        if appliedRecipe.map({batchRecipe != $0}) == true { learningRevisions[path,default:0]+=1;appliedRecipe=batchRecipe }
        if r==packaging.recipe { packaging.overrides[path]=nil }
        else { packaging.overrides[path]=r }
        storePackaging();saveQueue()
    }
    func usePackagingDefault() {
        guard let path=editingPackagePath,!recipeLocked else { return }
        leavePackagingEditor();clearAdjustments([path]);packaging.overrides[path]=nil;activatePhotoRecipe();storePackaging();saveQueue()
    }
    func savePackagingDefault() {
        guard let path=editingPackagePath,!recipeLocked else { return }
        var r=batchRecipe;r.package_cleanup=true;packaging.recipe=r;packaging.overrides[path]=nil
        storePackaging();saveQueue();schedulePreview()
    }
    static let learnedFields=["exposure","contrast","whites","warmth","sharpness","denoise","shadow"]
    static let photoFields=learnedFields+["fill","size"]
    func toneValues(_ r:BatchRecipe)->[String:Double] {
        ["exposure":r.exposure,"contrast":r.contrast,"whites":r.whites ?? 0,"warmth":r.warmth,"sharpness":r.sharpness,"denoise":r.denoise,"shadow":r.shadow,"fill":r.fill,"size":Double(r.size)]
    }
    func photoOptions(_ r:BatchRecipe)->[String:String] {
        ["shadow_method":r.shadow_method,"shadow_detection":r.shadow_detection ?? "dark","aspect":r.aspect,"framing":r.framing,"centered_scale":r.centered_scale ?? "shared","mask_mode":r.mask_mode,"mask_backend":r.mask_backend ?? "mps","small_parts":r.small_parts == true ? "true":"false","recover_parts":r.recover_parts == true ? "true":"false","recover_white":r.recover_white == true ? "true":"false","recover_dark":r.recover_dark == true ? "true":"false"]
    }
    func mergedOptions(_ r:BatchRecipe,_ values:[String:String])->BatchRecipe {
        var result=r
        result.shadow_method=values["shadow_method"] ?? r.shadow_method
        if let v=values["shadow_detection"] { result.shadow_detection=v=="dark" ? nil:v }
        result.aspect=values["aspect"] ?? r.aspect;result.framing=values["framing"] ?? r.framing
        if let v=values["centered_scale"] { result.centered_scale=v=="shared" ? nil:v }
        result.mask_mode=values["mask_mode"] ?? r.mask_mode
        if let v=values["mask_backend"] { result.mask_backend=v=="mps" ? nil:v }
        if let v=values["small_parts"] { result.small_parts=v=="true" ? true:nil }
        if let v=values["recover_parts"] { result.recover_parts=v=="true" ? true:nil }
        if let v=values["recover_white"] { result.recover_white=v=="true" ? true:nil }
        if let v=values["recover_dark"] { result.recover_dark=v=="true" ? true:nil }
        return result
    }
    func mergedTone(_ r:BatchRecipe,_ values:[String:Double])->BatchRecipe {
        var result=r
        result.exposure=values["exposure"] ?? r.exposure;result.contrast=values["contrast"] ?? r.contrast
        result.whites=values["whites"].map{$0==0 ? nil:$0} ?? r.whites
        result.warmth=values["warmth"] ?? r.warmth;result.sharpness=values["sharpness"] ?? r.sharpness
        result.denoise=values["denoise"] ?? r.denoise;result.shadow=values["shadow"] ?? r.shadow
        result.fill=values["fill"] ?? r.fill;result.size=values["size"].map{Int($0)} ?? r.size
        return result
    }
    func adjustedRecipe(_ base:BatchRecipe,path:String,job:BatchJob?=nil)->BatchRecipe {
        let adjustment=adjustmentFor(path,job:job)
        return mergedOptions(mergedTone(base,adjustment?.values ?? [:]),adjustment?.options ?? [:])
    }
    func adjustmentFor(_ path:String,job:BatchJob?=nil)->PhotoAdjustment? {
        let context=job ?? queue.jobs.first{$0.id==previewFolderID && $0.sources.contains(path)}
        if let context=context,context.started_recipe != nil { return context.adjustment_recipes?[path] }
        return photoAdjustments[path]
    }
    func adjustmentsFor(_ paths:[String])->[String:PhotoAdjustment] {
        Dictionary(uniqueKeysWithValues:paths.compactMap { path in adjustmentFor(path).map{(path,$0)} })
    }
    func clearStaleAdjustments(_ paths:[String]) {
        for path in paths {
            guard let a=photoAdjustments[path],let size=a.source_size,let modified=a.source_mtime,
                  let attrs=try? FileManager.default.attributesOfItem(atPath:path),
                  let actualSize=attrs[.size] as? NSNumber,let actualDate=attrs[.modificationDate] as? Date else { continue }
            if actualSize.int64Value != size || abs(actualDate.timeIntervalSince1970-modified)>0.001 {
                photoAdjustments[path]=nil;learningRevisions[path,default:0]+=1
            }
        }
    }
    func loadAdjustments() {
        guard FileManager.default.fileExists(atPath:adjustmentsURL.path) else { return }
        do { photoAdjustments=try JSONDecoder().decode([String:PhotoAdjustment].self,from:Data(contentsOf:adjustmentsURL)) }
        catch { adjustmentsAvailable=false;self.error="Could not read personalized settings. The saved file has been kept." }
    }
    func storeAdjustments() {
        guard adjustmentsAvailable else { return }
        do {
            try FileManager.default.createDirectory(at:adjustmentsURL.deletingLastPathComponent(),withIntermediateDirectories:true)
            try JSONEncoder().encode(photoAdjustments).write(to:adjustmentsURL,options:.atomic)
        } catch { self.error="Could not save personalized settings: "+error.localizedDescription }
    }
    func clearAdjustments(_ paths:[String]) {
        for path in paths { photoAdjustments[path]=nil;learningRevisions[path,default:0]+=1 }
        storeAdjustments()
    }
    func saveCurrentAdjustment() {
        guard let path=editingAdjustmentPath,!running,!queueRunning,!recipeLocked,
              let previous=appliedRecipe,batchRecipe != previous else { return }
        let values=toneValues(batchRecipe).filter { toneValues(previous)[$0.key] != $0.value }
        let options=photoOptions(batchRecipe).filter { photoOptions(previous)[$0.key] != $0.value }
        let targets=checked.map(\.path),all = !inputs.isEmpty && targets.count==inputs.count
        if all {
            regularRecipe=mergedOptions(mergedTone(regularRecipe,values),options)
            if let id=previewFolderID,let i=queue.jobs.firstIndex(where:{$0.id==id}) { queue.jobs[i].override=regularRecipe }
        }
        recordPhotoEdits(targets,values:values,options:options,inherit:all)
        storeAdjustments();storePackaging();rememberFolderRecipes();saveQueue()
        // The viewed photo can be unchecked. Always display its actual result.
        applyRecipe(adjustedRecipe(packagingRecipeFor(path) ?? regularRecipe,path:path))
    }
    func recordPhotoEdits(_ paths:[String],values:[String:Double],options:[String:String],inherit:Bool) {
        guard !values.isEmpty || !options.isEmpty else { return }
        for path in paths {
            var a=photoAdjustments[path] ?? PhotoAdjustment(values:[:])
            for (key,value) in values {
                a.values[key]=inherit ? nil:value
                if Self.learnedFields.contains(key),!a.manual_fields.contains(key) { a.manual_fields.append(key) }
            }
            var opts=a.options ?? [:]
            for (key,value) in options { opts[key]=inherit ? nil:value }
            a.options=opts.isEmpty ? nil:opts;a.manually_edited=true
            if inherit,let base=packagingRecipeFor(path) { packaging.overrides[path]=mergedOptions(mergedTone(base,values),options) }
            photoAdjustments[path]=a;learningRevisions[path,default:0]+=1
        }
    }
    func requestLearning(_ paths:[String]) {
        guard !running,!queueRunning,adjustmentsAvailable else { return }
        clearStaleAdjustments(paths)
        learningPending.formUnion(paths.filter { photoAdjustments[$0]?.manual_fields.isEmpty != false && photoAdjustments[$0]?.manually_edited != true })
        beginLearning()
    }
    func beginLearning() {
        guard learningProcess==nil,!learningPending.isEmpty,!running,!queueRunning else { return }
        let paths=learningPending.sorted();learningPending=[];learningBusy=true
        learningCancelled=false;learningApplied=[];learningNote="Checking saved processing history…"
        let revisions=learningRevisions
        let token=UUID().uuidString
        let folder=root.appendingPathComponent("work/learning")
        let input=folder.appendingPathComponent("request-\(token).json"),output=folder.appendingPathComponent("result-\(token).json")
        let progress=folder.appendingPathComponent("progress-\(token).json")
        do {
            try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
            let items=paths.map { ["source":$0,"kind":packagingRecipeFor($0)==nil ? "product":"packaging"] }
            try JSONSerialization.data(withJSONObject:["photos":items]).write(to:input,options:.atomic)
            let task=Process();task.executableURL=root.appendingPathComponent(".venv/bin/python");task.currentDirectoryURL=root
            task.arguments=[root.appendingPathComponent("recipe_learning.py").path,"--request",input.path,"--output",output.path,"--root",root.path,"--progress",progress.path]
            task.standardOutput=FileHandle.nullDevice;task.standardError=FileHandle.nullDevice
            task.terminationHandler={ [self] ended in Task { @MainActor in
                defer {
                    self.learningTimer?.invalidate();self.learningTimer=nil
                    try? FileManager.default.removeItem(at:progress)
                    try? FileManager.default.removeItem(at:input);try? FileManager.default.removeItem(at:output)
                    self.learningProcess=nil;self.learningBusy=false;self.flashLearningNote();self.beginLearning()
                }
                if self.learningCancelled { self.learningNote="Matching cancelled. Completed suggestions are kept.";return }
                guard ended.terminationStatus==0,let data=try? Data(contentsOf:output),let report=try? JSONDecoder().decode(LearningResponse.self,from:data) else {
                    self.learningNote="History matching failed. Your existing settings are kept.";return
                }
                self.applyLearning(report,revisions:revisions)
                let count=report.adjustments.values.filter{!$0.values.isEmpty}.count
                self.learningNote="\(count) photos matched · \(paths.count-count-report.errors.count) kept their settings" + (report.errors.isEmpty ? "":" · \(report.errors.count) unavailable")
            } }
            learningProcess=task;try task.run()
            learningTimer=Timer.scheduledTimer(withTimeInterval:1,repeats:true) { [self] _ in
                Task { @MainActor in
                    guard self.learningProcess === task,!self.learningCancelled,
                          let data=try? Data(contentsOf:progress),let report=try? JSONDecoder().decode(LearningResponse.self,from:data) else { return }
                    if let message=report.message { self.learningNote=message }
                    self.applyLearning(report,revisions:revisions)
                }
            }
        } catch { learningProcess=nil;learningBusy=false;learningNote="Could not start history matching: "+error.localizedDescription }
    }
    func applyLearning(_ report:LearningResponse,revisions:[String:Int]) {
        let fresh=report.adjustments.filter { !learningApplied.contains($0.key) }
        guard !fresh.isEmpty else { return }
        leavePackagingEditor()
        for (path,a) in fresh {
            learningApplied.insert(path)
            guard learningRevisions[path,default:0]==revisions[path,default:0],photoAdjustments[path]?.manual_fields.isEmpty != false,photoAdjustments[path]?.manually_edited != true else { continue }
            photoAdjustments[path]=a.values.isEmpty ? nil:a
        }
        storeAdjustments();saveQueue()
        if queueVisible,let id=selectedBatchID { selectBatch(id) }
        else { activatePhotoRecipe() }
        schedulePreview()
    }
    func cancelLearning() {
        learningPending=[];learningCancelled=true
        learningNote="Cancelling matching…"
        learningProcess?.terminate()
    }

    func loadFolderRecipes() {
        guard FileManager.default.fileExists(atPath:folderRecipesURL.path) else { return }
        do { folderRecipes=try JSONDecoder().decode([String:BatchRecipe].self,from:Data(contentsOf:folderRecipesURL)) }
        catch { folderRecipesAvailable=false;self.error="Could not read remembered folder settings. The saved file has been kept." }
    }
    func rememberFolderRecipes() {
        guard folderRecipesAvailable else { return }
        for job in queue.jobs where job.started_recipe==nil && job.archived != true {
            folderRecipes[job.folder]=job.override ?? queue.shared_recipe
        }
        do {
            try FileManager.default.createDirectory(at:folderRecipesURL.deletingLastPathComponent(),withIntermediateDirectories:true)
            try JSONEncoder().encode(folderRecipes).write(to:folderRecipesURL,options:.atomic)
        } catch { self.error="Could not remember folder settings: "+error.localizedDescription }
    }
    func batchPreviewRecipe(_ job:BatchJob)->BatchRecipe {
        let source=previewSource(job) ?? ""
        let base=(source.isEmpty ? nil:packagingRecipeFor(source,job:job)) ?? job.started_recipe ?? job.override ?? queue.shared_recipe
        return adjustedRecipe(base,path:source,job:job)
    }
    func batchSignature(_ job:BatchJob)->String {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        let r=batchPreviewRecipe(job)
        return String(data:(try? encoder.encode(r)) ?? Data(),encoding:.utf8)! + (previewSource(job) ?? "") + (r.framing=="both" ? previewFraming:"")
    }
    func batchThumbnail(_ path:String)->NSImage? {
        guard let source=CGImageSourceCreateWithURL(URL(fileURLWithPath:path) as CFURL,nil),
              let image=CGImageSourceCreateThumbnailAtIndex(source,0,[kCGImageSourceCreateThumbnailFromImageAlways:true,kCGImageSourceThumbnailMaxPixelSize:640,kCGImageSourceCreateThumbnailWithTransform:true] as CFDictionary) else { return nil }
        return NSImage(cgImage:image,size:NSSize(width:image.width,height:image.height))
    }
    func loadCompletedBatchImages() {
        for job in activeBatches where job.started_recipe != nil {
            let signature="export:"+batchSignature(job)+":"+job.status
            let needsDetail=job.id==selectedBatchID && (batchDetailJobID != job.id || batchDetailSignature != signature)
            let wanted=previewSource(job)
            guard batchPreviewSignatures[job.id] != signature || needsDetail,let metadata=job.metadata,
                  let data=try? Data(contentsOf:URL(fileURLWithPath:metadata).appendingPathComponent("manifest.json")),
                  let manifest=try? JSONDecoder().decode(Manifest.self,from:data),let photo=manifest.records.first(where:{$0.source==wanted}) ?? manifest.records.first else { continue }
            let recipe=job.started_recipe ?? job.override ?? queue.shared_recipe
            let variant=recipe.framing=="both" ? previewFraming:recipe.framing
            let imagePath=photo.variants?[variant]?.jpeg ?? photo.jpeg
            guard let image=batchThumbnail(imagePath) else { continue }
            batchImages[job.id]=image;batchPreviewSignatures[job.id]=signature
            batchPreviewNotes[job.id]="Exported result"
            if needsDetail,let full=NSImage(contentsOfFile:imagePath) {
                batchDetailImage=full;batchDetailJobID=job.id;batchDetailSignature=signature
            }
        }
    }
    func needsBatchPreview(_ job:BatchJob)->Bool {
        guard job.archived != true,!job.sources.isEmpty,job.started_recipe==nil,batchPreviewFailures[job.id] != batchSignature(job) else { return false }
        return batchPreviewSignatures[job.id] != batchSignature(job) ||
            (job.id==selectedBatchID && (batchDetailJobID != job.id || batchDetailSignature != batchSignature(job)))
    }
    func scheduleBatchPreviews() {
        guard queueVisible,!queueRunning,!running else { return }
        previewWork?.cancel()
        let work=DispatchWorkItem { [weak self] in self?.sendNextBatchPreview() }
        previewWork=work;DispatchQueue.main.asyncAfter(deadline:.now()+0.25,execute:work)
    }
    // A single front view understates a batch's shared scale whenever a side
    // rotation is wider, so the tenth frame (roughly a 90° turn on this
    // pipeline's 10°-step turntable) is checked too: between a straight-on
    // and a broadside view, one of the two reliably bounds the true extent.
    func wideReference(for job:BatchJob)->String? {
        guard job.sources.count>1 else { return nil }
        return job.sources.count>9 ? job.sources[9] : job.sources.last
    }
    func sendNextBatchPreview() {
        guard queueVisible,!queueRunning,!running,previewEnabled,batchPreviewActive==nil else { return }
        loadCompletedBatchImages()
        let jobs=activeBatches.sorted { ($0.id==selectedBatchID ? 0:1)<($1.id==selectedBatchID ? 0:1) }
        guard let job=jobs.first(where:needsBatchPreview),
              let source=previewSource(job) else { return }
        let signature=batchSignature(job)
        if let wideRef=wideReference(for:job),wideRef != source,knownFrames[wideRef]==nil,batchProbeFailures[job.id] != signature {
            sendGeometryProbe(wideRef,job:job,signature:signature);return
        }
        let r=batchPreviewRecipe(job)
        var settings=(try? JSONSerialization.jsonObject(with:JSONEncoder().encode(r))) as? [String:Any] ?? [:]
        settings["package_cleanup"]=packagingRecipeFor(source,job:job) != nil
        settings["small_parts"]=r.small_parts ?? false
        settings["recover_parts"]=r.recover_parts ?? false
        settings["recover_white"]=r.recover_white ?? false
        settings["recover_dark"]=r.recover_dark ?? false
        if r.framing=="both" { settings["framing"]=previewFraming }
        previewSerial+=1
        batchPreviewActive=(job.id,signature,previewSerial,false)
        batchPreviewDetailSerial=job.id==selectedBatchID ? previewSerial:nil
        batchPreviewNotes[job.id]="Preparing preview…"
        let request:[String:Any]=["id":previewSerial,"source":source,"settings":settings,"mask_mode":r.mask_mode,"mask_backend":r.mask_backend ?? "mps","thumbnail":job.id != selectedBatchID,
                                  "batch_geometry":job.sources.filter { packagingRecipeFor(source,job:job) != nil ? $0==source:packagingRecipeFor($0,job:job)==nil }.compactMap { knownFrames[$0] }]
        do {
            var bytes=try JSONSerialization.data(withJSONObject:request);bytes.append(10)
            try previewInput?.write(contentsOf:bytes)
        } catch {
            batchPreviewNotes[job.id]="Preview unavailable: "+error.localizedDescription
            batchPreviewFailures[job.id]=signature;batchPreviewActive=nil;scheduleBatchPreviews()
        }
    }
    func sendGeometryProbe(_ source:String,job:BatchJob,signature:String) {
        let r=adjustedRecipe(packagingRecipeFor(source,job:job) ?? job.override ?? queue.shared_recipe,path:source,job:job)
        previewSerial+=1
        batchPreviewActive=(job.id,signature,previewSerial,true)
        batchPreviewNotes[job.id]="Checking widest angle…"
        let request:[String:Any]=["id":previewSerial,"source":source,"geometry_only":true,
                                  "mask_mode":r.mask_mode,"mask_backend":r.mask_backend ?? "mps",
                                  "settings":["small_parts":r.small_parts ?? false,"recover_parts":r.recover_parts ?? false,"recover_white":r.recover_white ?? false,"recover_dark":r.recover_dark ?? false]]
        do {
            var bytes=try JSONSerialization.data(withJSONObject:request);bytes.append(10)
            try previewInput?.write(contentsOf:bytes)
        } catch {
            // A failed probe should not stall the batch forever; the main
            // preview still renders, just with whatever scale is known so far.
            batchProbeFailures[job.id]=signature;batchPreviewActive=nil;scheduleBatchPreviews()
        }
    }
    func consumeGeometryProbe(_ event:[String:Any],active:(id:String,signature:String,serial:Int,isProbe:Bool)) {
        let kind=event["event"] as? String
        if kind=="preview_progress" { batchPreviewNotes[active.id]=(event["phase"] as? String ?? "Checking widest angle")+"…" }
        // Mask-building for the probe emits the same progress events as a real
        // render; only its own terminal event should end the probe.
        guard kind=="geometry_result" || kind=="preview_error" else { return }
        batchPreviewActive=nil
        if kind=="geometry_result",let source=event["source"] as? String,let box=event["bbox"] as? [Int],let size=event["source_size"] as? [Int] {
            knownFrames[source]=["source":source,"bbox":box,"source_size":size]
        } else {
            // A probe that errors would otherwise be retried on every
            // subsequent schedule call; give up on it for this batch signature
            // so the main preview can still proceed.
            batchProbeFailures[active.id]=active.signature
        }
        scheduleBatchPreviews()
    }
    func consumeBatchPreview(_ event:[String:Any],active:(id:String,signature:String,serial:Int,isProbe:Bool)) {
        let kind=event["event"] as? String
        if kind=="preview_progress" { batchPreviewNotes[active.id]=(event["phase"] as? String ?? "Preparing preview")+"…" }
        guard kind=="preview_result" || kind=="preview_error" else { return }
        let isDetail=batchPreviewDetailSerial==active.serial
        batchPreviewActive=nil;batchPreviewDetailSerial=nil
        if let job=queue.jobs.first(where:{$0.id==active.id}),batchSignature(job)==active.signature {
            if kind=="preview_result",let path=event["path"] as? String,let image=batchThumbnail(path) {
                batchImages[active.id]=image;batchPreviewSignatures[active.id]=active.signature
                if isDetail,selectedBatchID==active.id,let bytes=try? Data(contentsOf:URL(fileURLWithPath:path)),let full=NSImage(data:bytes) {
                    batchDetailImage=full;batchDetailJobID=active.id;batchDetailSignature=active.signature
                }
                batchPreviewFailures[active.id]=nil
                batchPreviewNotes[active.id]="Live recipe · framing provisional"
                if let reviews=event["review"] as? [String],!reviews.isEmpty { batchPreviewNotes[active.id]="Review: "+reviews.joined(separator:". ") }
                if let source=event["source"] as? String,let box=event["bbox"] as? [Int],let size=event["source_size"] as? [Int] {
                    knownFrames[source]=["source":source,"bbox":box,"source_size":size]
                }
                // The renderer owns bounded persistent previews, including this file.
            } else {
                batchPreviewNotes[active.id]="Preview unavailable: \(event["message"] as? String ?? "Unable to read image")"
                batchPreviewFailures[active.id]=active.signature
            }
        }
        if queueVisible { scheduleBatchPreviews() } else { schedulePreview() }
    }
    func retryBatchPreview(_ id:String) {
        batchPreviewFailures[id]=nil;batchPreviewSignatures[id]=nil;batchProbeFailures[id]=nil
        if !queueRunning { startPreviewWorker();scheduleBatchPreviews() }
    }
    func chooseQueueDestination() {
        let panel=NSOpenPanel();panel.canChooseFiles=false;panel.canChooseDirectories=true;panel.canCreateDirectories=true
        if panel.runModal() == .OK,let url=panel.url { queue.output_root=url.path;saveQueue() }
    }
    /// A paused run can continue: a selected folder is paused, so Process becomes Resume.
    var canResume:Bool { queue.status=="paused" && exportableBatches.contains { $0.status=="paused" } }
    func resumeQueue() { startQueue(resume:true) }
    func startQueue(resume:Bool=false) {
        guard !learningBusy else { return }
        guard !running,!queueRunning,!exportableBatches.isEmpty else { return }
        // Finished / stopped / error batches export again into the same folder when
        // that export still exists; otherwise a new folder is created. Resuming leaves
        // folders finished earlier in the run alone, and the worker skips them.
        for i in queue.jobs.indices where !resume && queue.jobs[i].archived != true && queue.jobs[i].include != false && ["complete","error","stopped"].contains(queue.jobs[i].status) {
            let job=queue.jobs[i]
            let outPath=job.output ?? ""
            let metaPath=job.metadata ?? ""
            let canReplace = !outPath.isEmpty && !metaPath.isEmpty
                && FileManager.default.fileExists(atPath:outPath)
                && FileManager.default.fileExists(atPath:(metaPath as NSString).appendingPathComponent("manifest.json"))
            queue.jobs[i].override=job.started_recipe ?? job.override
            queue.jobs[i].started_recipe=nil
            queue.jobs[i].status="queued";queue.jobs[i].message=nil
            queue.jobs[i].completed=0;queue.jobs[i].errors=0;queue.jobs[i].attempts=0
            if canReplace {
                queue.jobs[i].replace=true
            } else {
                let run=UUID().uuidString.prefix(8)
                queue.jobs[i].replace=nil
                queue.jobs[i].output=URL(fileURLWithPath:queue.output_root).appendingPathComponent("\(job.name)-\(run)").path
                queue.jobs[i].metadata=root.appendingPathComponent("work/shoots/queue-\(job.id)-\(run)").path
            }
        }
        try? FileManager.default.removeItem(at:queueURL.deletingLastPathComponent().appendingPathComponent("pause.request"))
        try? FileManager.default.removeItem(at:queueURL.deletingLastPathComponent().appendingPathComponent("stop.request"))
        // Elapsed time restarts with every press of Process; Resume continues it without the paused time.
        let carried=resume ? queueElapsed().map { $0 } ?? 0 : 0
        queue.run_started=Date().timeIntervalSince1970-carried;queue.run_ended=nil;queuePhotoKey="";queuePhotoStart=nil
        pauseRequested=false
        runSeen=true;stopPreviewWorker();queue.status="starting";queue.phase="Preparing queue";saveQueue()
        do {
            let task=Process();task.executableURL=root.appendingPathComponent(".venv/bin/python");task.currentDirectoryURL=root
            task.arguments=[root.appendingPathComponent("queue_worker.py").path,queueURL.path]
            let log=queueURL.deletingLastPathComponent().appendingPathComponent("worker.log")
            FileManager.default.createFile(atPath:log.path,contents:nil)
            let handle=try FileHandle(forWritingTo:log);task.standardOutput=handle;task.standardError=handle
            task.terminationHandler={ [self] _ in Task { @MainActor in
                try? handle.close();self.pollQueue();self.queueProcess=nil
                if self.queueRunning { self.queue.status="paused";self.queue.phase="Worker stopped. Review worker.log, then resume.";self.saveQueue() }
            } }
            queueProcess=task;try task.run();watchQueue()
        } catch { queue.status="paused";queue.phase=error.localizedDescription;saveQueue();self.error=error.localizedDescription }
    }
    func watchQueue() {
        queueTimer?.invalidate()
        queueTimer=Timer.scheduledTimer(withTimeInterval:0.5,repeats:true) { [self] _ in Task { @MainActor in self.pollQueue() } }
    }
    func pollQueue() {
        guard let data=try? Data(contentsOf:queueURL),var object=(try? JSONSerialization.jsonObject(with:data)) as? [String:Any] else { return }
        if object["status"] as? String == "running",
           let liveData=try? Data(contentsOf:queueURL.deletingLastPathComponent().appendingPathComponent("progress.json")),
           let live=(try? JSONSerialization.jsonObject(with:liveData)) as? [String:Any],
           let pid=object["worker_pid"] as? Int,live["worker_pid"] as? Int == pid,
           let active=object["active_job_id"] as? String,live["active_job_id"] as? String == active,
           let checkpointDate=(try? FileManager.default.attributesOfItem(atPath:queueURL.path))?[.modificationDate] as? Date,
           let progressDate=(try? FileManager.default.attributesOfItem(atPath:queueURL.deletingLastPathComponent().appendingPathComponent("progress.json").path))?[.modificationDate] as? Date,
           progressDate >= checkpointDate {
            for (key,value) in live where key != "active_completed" { object[key]=value }
            let photo="\(active):\(live["photo_number"] as? Int ?? 0)"
            if photo != queuePhotoKey {
                queuePhotoKey=photo;queuePhotoStart=Date().addingTimeInterval(-(live["photo_elapsed"] as? Double ?? 0))
            }
            if var jobs=object["jobs"] as? [[String:Any]],let i=jobs.firstIndex(where:{$0["id"] as? String == active}) {
                jobs[i]["completed"]=live["active_completed"];object["jobs"]=jobs
            }
        }
        guard let merged=try? JSONSerialization.data(withJSONObject:object),var saved=try? JSONDecoder().decode(BatchQueueState.self,from:merged) else { return }
        if saved.status=="running",let pid=saved.worker_pid,Darwin.kill(pid_t(pid),0) != 0 { saved.status="paused";saved.phase="Interrupted · resume saved progress" }
        queue=saved
        if queueRunning { runSeen=true }
        loadCompletedBatchImages()
        if !queueRunning {
            queuePhotoKey="";queuePhotoStart=nil;pauseRequested=false
            if queue.run_started != nil && queue.run_ended==nil { queue.run_ended=Date().timeIntervalSince1970;saveQueue() }
            settleFinishedBatches()
            queueTimer?.invalidate();queueTimer=nil
            if queueVisible { startPreviewWorker();scheduleBatchPreviews() }
        }
    }
    /// Real time since Process was pressed; frozen once that run ends. nil when unknown.
    var activeQueueJob:BatchJob? {
        guard queueRunning || ["paused","complete","stopped"].contains(queue.status) else { return nil }
        if let id=queue.active_job_id { return queue.jobs.first{$0.id==id} }
        return queue.jobs.first{$0.status=="running"}
    }
    var activeBatchProgressText:String {
        guard let job=activeQueueJob else { return queue.phase ?? "Preparing queue…" }
        let total=job.sources.count,done=min(total,max(0,job.completed))
        let percentage=total>0 ? Int(Double(done)*100/Double(total)):0
        var parts:[String]=[]
        let photoTotal=queue.photo_total ?? total
        if queueRunning,let number=queue.photo_number,number>0,photoTotal>0 {
            parts.append("\(queue.photo_stage=="preparation" ? "Preparing photo":"Photo") \(min(number,photoTotal)) of \(photoTotal)")
        }
        parts.append("\(done)/\(total) exported · \(percentage)%")
        if job.errors>0 { parts.append("\(job.errors) failed") }
        return parts.joined(separator:" · ")
    }
    /// Folders in the current run: selection cannot change while it runs.
    var runJobs:[BatchJob] { activeBatches.filter { $0.include != false } }
    /// Photos finished in a folder: exported ones, or all of them once the folder is done or has given up.
    func finishedPhotos(_ job:BatchJob)->Int {
        ["complete","error"].contains(job.status) ? job.sources.count:min(job.sources.count,max(0,job.completed))
    }
    var runPhotoTotal:Int { runJobs.reduce(0) { $0+$1.sources.count } }
    var runPhotoDone:Int { runJobs.reduce(0) { $0+finishedPhotos($1) } }
    var runProgress:Double { runPhotoTotal==0 ? 0:Double(runPhotoDone)/Double(runPhotoTotal) }
    var runFoldersLeft:Int { runJobs.filter { !["complete","error"].contains($0.status) }.count }
    var runFolderPosition:String? {
        guard let id=activeQueueJob?.id,let index=runJobs.firstIndex(where:{$0.id==id}) else { return nil }
        return "Folder \(index+1) of \(runJobs.count)"
    }
    func queueElapsed(_ now:Date=Date())->Double? {
        guard let start=queue.run_started else { return nil }
        if queueRunning { return max(0,now.timeIntervalSince1970-start) }
        return queue.run_ended.map { max(0,$0-start) }
    }
    /// Real time on the photo being processed, started from the worker's last report.
    func queuePhotoElapsed(_ now:Date=Date())->Double {
        guard let start=queuePhotoStart else { return queue.photo_elapsed ?? 0 }
        return max(0,now.timeIntervalSince(start))
    }
    func pauseQueue() {
        pauseRequested=true
        try? Data().write(to:queueURL.deletingLastPathComponent().appendingPathComponent("pause.request"))
    }
    func stopQueue() {
        try? Data().write(to:queueURL.deletingLastPathComponent().appendingPathComponent("stop.request"))
        // Read the live pid so the photo being rendered ends now, not when it finishes.
        if let data=try? Data(contentsOf:queueURL),let live=try? JSONDecoder().decode(BatchQueueState.self,from:data),
           live.worker_pid != nil,let child=live.child_pid { Darwin.kill(pid_t(child),SIGTERM) }
    }
    /// Runs whenever no queue is running: returns batches to how they were before Process,
    /// apart from exports already delivered, which are never removed.
    func settleFinishedBatches() {
        var changed=false
        for i in queue.jobs.indices where queue.jobs[i].status=="complete" {
            guard let exported=queue.jobs[i].started_recipe else { continue }
            // Re-processing always exports into a new folder, so a finished batch needs no lock.
            // Keep the exact recipe it was exported with as its folder recipe.
            if !(queue.jobs[i].override==nil && exported==queue.shared_recipe) { queue.jobs[i].override=exported }
            queue.jobs[i].started_recipe=nil
            changed=true
        }
        for i in queue.jobs.indices where queue.jobs[i].status=="stopped" {
            let job=queue.jobs[i]
            // Reversible: the unfinished export goes to the Trash, never deleted outright.
            if let path=job.output,FileManager.default.fileExists(atPath:path) {
                guard (try? FileManager.default.trashItem(at:URL(fileURLWithPath:path),resultingItemURL:nil)) != nil else { continue }
            }
            guard job.started_recipe != nil || job.output != nil || job.completed != 0 else { continue }
            // Nothing was delivered, so unfreeze the recipe as if the batch never started.
            queue.jobs[i].started_recipe=nil;queue.jobs[i].output=nil;queue.jobs[i].metadata=nil
            queue.jobs[i].completed=0;queue.jobs[i].errors=0;queue.jobs[i].attempts=0
            changed=true
        }
        if changed { saveQueue() }
    }
    func retryFailedBatches() {
        guard !queueRunning else { return }
        for i in queue.jobs.indices where queue.jobs[i].archived != true && queue.jobs[i].status=="error" { queue.jobs[i].status="paused";queue.jobs[i].attempts=0;queue.jobs[i].message=nil }
        queue.status="paused";saveQueue()
    }
    func openSummary() { if let path=queue.summary { NSWorkspace.shared.open(URL(fileURLWithPath:path)) } }
    var selectedBatchCount:Int { exportableBatches.count }
    var exportableBatches:[BatchJob] { activeBatches.filter { $0.include != false } }
    var reexportsFinishedBatches:Bool { exportableBatches.contains { ["complete","error","stopped"].contains($0.status) } }
    func setBatchIncluded(_ id:String,_ value:Bool) {
        guard !queueRunning,!running,let i=queue.jobs.firstIndex(where:{$0.id==id && $0.archived != true}) else { return }
        queue.jobs[i].include=value;batchSelectionAnchor=id;saveQueue()
    }
    func selectAllBatches() {
        guard !queueRunning else { return }
        for i in queue.jobs.indices where queue.jobs[i].archived != true { queue.jobs[i].include=true };batchSelectionAnchor=activeBatches.first?.id;saveQueue()
    }
    func clearBatchSelection() {
        guard !queueRunning else { return }
        for i in queue.jobs.indices { queue.jobs[i].include=false };batchSelectionAnchor=nil;saveQueue()
    }

    func clickBatch(_ id:String,shift:Bool=false,command:Bool=false) {
        guard !queueRunning,!running,let index=activeBatches.firstIndex(where:{$0.id==id}) else { return }
        var chosen=Set(exportableBatches.map(\.id))
        if shift,let anchor=batchSelectionAnchor,let start=activeBatches.firstIndex(where:{$0.id==anchor}) {
            let range=Set(activeBatches[min(start,index)...max(start,index)].map(\.id))
            chosen=command ? chosen.union(range):range
        } else if command {
            if chosen.contains(id) { chosen.remove(id) } else { chosen.insert(id) }
            batchSelectionAnchor=id
        } else { chosen=[id];batchSelectionAnchor=id }
        for i in queue.jobs.indices where queue.jobs[i].archived != true { queue.jobs[i].include=chosen.contains(queue.jobs[i].id) }
        selectBatch(id);saveQueue()
    }
    /// A folder action applies to every selected folder when the one acted on is part of
    /// that selection, and to just that folder otherwise.
    func actionTargets(_ id:String)->[String] {
        let selected=exportableBatches.map(\.id)
        return selected.count>1 && selected.contains(id) ? selected:[id]
    }
    func archiveBatches(_ ids:[String]) {
        guard !queueRunning,!running else { return }
        let ids=Set(ids)
        for i in queue.jobs.indices where ids.contains(queue.jobs[i].id) {
            queue.jobs[i].archived=true;queue.jobs[i].include=false
        }
        // Completed folders collect at the end, so new imports can join the end of the working row.
        queue.jobs=queue.jobs.filter{$0.archived != true}+queue.jobs.filter{$0.archived == true}
        if let active=batchPreviewActive,ids.contains(active.id) { stopPreviewWorker() }
        if let id=selectedBatchID,ids.contains(id) {
            selectedBatchID=nil;batchDetailImage=nil;batchDetailJobID=nil;batchDetailSignature=""
            if let next=activeBatches.first { selectBatch(next.id) }
            else { recipeTarget=nil;applyRecipe(queue.shared_recipe) }
        }
        batchSelectionAnchor=nil;draggedBatchIDs=[];saveQueue()
        if !activeBatches.isEmpty { startPreviewWorker();scheduleBatchPreviews() }
    }
    func restoreBatch(_ id:String) {
        guard !queueRunning,!running,let index=queue.jobs.firstIndex(where:{$0.id==id}) else { return }
        var job=queue.jobs.remove(at:index);job.archived=nil;job.include=true
        // Restore at the end of the working row, keeping its recipe and exports.
        let position=queue.jobs.firstIndex(where:{$0.archived==true}) ?? queue.jobs.count
        queue.jobs.insert(job,at:position);completedVisible=false;saveQueue();selectBatch(id)
        startPreviewWorker();scheduleBatchPreviews()
    }
    func moveBatches(_ ids:[String],before target:String?) {
        guard !queueRunning,!running else { return }
        let moving=Set(ids).intersection(Set(activeBatches.map(\.id)))
        guard !moving.isEmpty,target.map({!moving.contains($0)}) ?? true else { return }
        let block=activeBatches.filter { moving.contains($0.id) }
        var rest=activeBatches.filter { !moving.contains($0.id) }
        let index=target.flatMap { id in rest.firstIndex(where:{$0.id==id}) } ?? rest.count
        rest.insert(contentsOf:block,at:index)
        queue.jobs=rest+completedBatches;saveQueue()
    }
    /// Opens a batch's export. A moved folder is found again by name under the export
    /// destination and the queue is corrected; a missing one says so instead of doing nothing.
    func resolvedExport(_ id:String)->URL? {
        guard let i=queue.jobs.firstIndex(where:{$0.id==id}),let path=queue.jobs[i].output else { return nil }
        let manager=FileManager.default
        if manager.fileExists(atPath:path) { return URL(fileURLWithPath:path) }
        // Moved with the export destination: find it again by name and correct the queue.
        let moved=URL(fileURLWithPath:queue.output_root).appendingPathComponent(URL(fileURLWithPath:path).lastPathComponent)
        if manager.fileExists(atPath:moved.path) { queue.jobs[i].output=moved.path;saveQueue();return moved }
        error="The export folder for \(queue.jobs[i].name) is no longer at:\n\(path)\n\nIf you moved it somewhere else, open it from there. The batch keeps its recipe either way."
        return nil
    }
    func openExport(_ id:String) { if let url=resolvedExport(id) { NSWorkspace.shared.open(url) } }
    func batchDragProvider(_ id:String)->NSItemProvider {
        guard !queueRunning,!running else { return NSItemProvider() }
        if !exportableBatches.contains(where:{$0.id==id}) { clickBatch(id) }
        draggedBatchIDs=exportableBatches.map(\.id)
        let bytes=(try? JSONEncoder().encode(draggedBatchIDs)) ?? Data()
        let provider=NSItemProvider()
        provider.registerDataRepresentation(forTypeIdentifier:Self.batchDragType,visibility:.ownProcess) { completion in completion(bytes,nil);return nil }
        provider.suggestedName=draggedBatchIDs.count>1 ? "\(draggedBatchIDs.count) folders":queue.jobs.first{$0.id==id}?.name
        return provider
    }
    static func duration(_ seconds:Double)->String {
        let value=max(0,Int(seconds.rounded()))
        return value>=3600 ? String(format:"%d:%02d:%02d",value/3600,(value/60)%60,value%60) : String(format:"%02d:%02d",value/60,value%60)
    }
}

struct BatchOrderDrop:DropDelegate {
    let studio:Studio
    let target:String?
    func validateDrop(info:DropInfo)->Bool { !studio.queueRunning && !studio.running && !studio.draggedBatchIDs.isEmpty && info.hasItemsConforming(to:[Studio.batchDragType]) }
    func dropEntered(info:DropInfo) { studio.batchDropTarget=target ?? "end" }
    func dropExited(info:DropInfo) { studio.batchDropTarget=nil }
    func dropUpdated(info:DropInfo)->DropProposal? { DropProposal(operation:.move) }
    func performDrop(info:DropInfo)->Bool {
        guard validateDrop(info:info) else { return false }
        var before=target
        if let target=target,info.location.x>74,let index=studio.activeBatches.firstIndex(where:{$0.id==target}) {
            before=studio.activeBatches.dropFirst(index+1).first { !studio.draggedBatchIDs.contains($0.id) }?.id
        }
        if let target=target,studio.draggedBatchIDs.contains(target) { /* Dropping on the selected block leaves it in place. */ }
        else { studio.moveBatches(studio.draggedBatchIDs,before:before) }
        studio.draggedBatchIDs=[];studio.batchDropTarget=nil;return true
    }
}

let appBackground=Color(white:0.11)
let panelBackground=Color(white:0.135)
let stageBackground=Color(white:0.17)
let accent=Color(red:0.36,green:0.43,blue:0.96)
let muted=Color(white:0.65)

extension BatchJob {
    /// "set-11012-1-model-raw-photos" reads as "11012-1"; the full name stays in the tooltip.
    var shortName:String {
        if let r=name.range(of:#"(?<=set-)[^-]+-\d+"#,options:.regularExpression) { return String(name[r]) }
        return name
    }
}

struct PhotoView:View {
    let path:String
    var body:some View {
        if let image=NSImage(contentsOfFile:path) { Image(nsImage:image).resizable().aspectRatio(contentMode:.fit) }
        else { Image(systemName:"photo").font(.system(size:42)).foregroundStyle(muted) }
    }
}

/// Quiet icon/text button: a soft fill on hover, a stronger one while pressed.
struct ToolButtonStyle:ButtonStyle {
    var radius:CGFloat=6
    func makeBody(configuration:Configuration)->some View { ToolButtonBody(configuration:configuration,radius:radius) }
    struct ToolButtonBody:View {
        let configuration:ButtonStyleConfiguration
        let radius:CGFloat
        @Environment(\.isEnabled) private var enabled
        @State private var hovering=false
        var body:some View {
            configuration.label
                .background(RoundedRectangle(cornerRadius:radius).fill(Color.white.opacity(!enabled ? 0:configuration.isPressed ? 0.16:hovering ? 0.09:0)))
                .opacity(enabled ? 1:0.4)
                .contentShape(RoundedRectangle(cornerRadius:radius))
                .onHover { hovering=$0 }
                .animation(.easeOut(duration:0.12),value:hovering)
        }
    }
}
/// Filled accent button that brightens on hover and dims while pressed.
struct AccentButtonStyle:ButtonStyle {
    var radius:CGFloat=7
    func makeBody(configuration:Configuration)->some View { AccentButtonBody(configuration:configuration,radius:radius) }
    struct AccentButtonBody:View {
        let configuration:ButtonStyleConfiguration
        let radius:CGFloat
        @Environment(\.isEnabled) private var enabled
        @State private var hovering=false
        var body:some View {
            configuration.label
                .background(RoundedRectangle(cornerRadius:radius).fill(enabled ? accent:Color.white.opacity(0.14)))
                .overlay(RoundedRectangle(cornerRadius:radius).fill(Color.white.opacity(!enabled ? 0:configuration.isPressed ? 0:hovering ? 0.12:0)))
                .overlay(RoundedRectangle(cornerRadius:radius).fill(Color.black.opacity(enabled && configuration.isPressed ? 0.15:0)))
                .foregroundStyle(enabled ? Color.white:Color.white.opacity(0.45))
                .contentShape(RoundedRectangle(cornerRadius:radius))
                .onHover { hovering=$0 }
                .animation(.easeOut(duration:0.12),value:hovering)
        }
    }
}
/// Hover fill for views that are not Buttons (menus, tiles, rows).
struct HoverFill:ViewModifier {
    var radius:CGFloat
    var amount:Double
    @Environment(\.isEnabled) private var enabled
    @State private var hovering=false
    func body(content:Content)->some View {
        content.background(RoundedRectangle(cornerRadius:radius).fill(Color.white.opacity(enabled && hovering ? amount:0)))
            .onHover { hovering=$0 }
            .animation(.easeOut(duration:0.12),value:hovering)
    }
}
extension View {
    func hoverFill(_ radius:CGFloat=6,_ amount:Double=0.09)->some View { modifier(HoverFill(radius:radius,amount:amount)) }
}

/// One cell per photo of the folder being processed: written photos filled, the current one
/// filling with its progress, masks already prepared shown lighter.
struct PhotoSegments:View {
    let total:Int
    let written:Int
    let current:Int?
    let currentProgress:Double
    let preparing:Bool
    var body:some View {
        GeometryReader { geo in
            let gap:CGFloat=total>60 ? 0.5:1.5
            let width=max(1,(geo.size.width-gap*CGFloat(max(0,total-1)))/CGFloat(max(1,total)))
            HStack(spacing:gap) {
                ForEach(0..<max(0,total),id:\.self) { i in
                    let fill:Double = i<written ? 1:(i==current ? min(1,max(0.08,currentProgress)):(preparing && current.map { i<$0 } == true ? 1:0))
                    let color:Color = i<written ? accent:(preparing ? Color.white.opacity(0.45):accent)
                    ZStack(alignment:.leading) {
                        RoundedRectangle(cornerRadius:1).fill(Color.white.opacity(0.12))
                        RoundedRectangle(cornerRadius:1).fill(color).frame(width:width*fill)
                    }.frame(width:width)
                }
            }
        }.frame(height:6)
    }
}

enum HoverZoom { static let levels:[Double]=[2,3,4,6,8] }

/// Turns scroll-wheel movement over the photo into zoom steps; only listens while the pointer is on the photo.
final class ScrollZoomMonitor:ObservableObject {
    var hovering=false
    var onStep:((Int)->Void)?
    private var token:Any?
    private var carry:CGFloat=0
    func start() {
        guard token==nil else { return }
        token=NSEvent.addLocalMonitorForEvents(matching:.scrollWheel) { [weak self] event in
            guard let self=self,self.hovering else { return event }
            if !event.momentumPhase.isEmpty { return nil }
            self.carry += event.hasPreciseScrollingDeltas ? event.scrollingDeltaY:event.scrollingDeltaY*20
            if abs(self.carry)>=20 {
                self.onStep?(self.carry>0 ? 1:-1);self.carry=0
            }
            return nil
        }
    }
    func stop() { if let token=token { NSEvent.removeMonitor(token) };token=nil }
    deinit { stop() }
}

struct ZoomPhotoView:View {
    let path:String
    let image:NSImage?
    let pixels:CGImage?
    @State private var pointer:CGPoint?
    /// Magnification of the loupe; 0 means actual pixels. Remembered between sessions.
    @AppStorage("hoverZoomLevel") private var level=2.0
    @AppStorage("hoverZoomOn") private var zoomOn=true
    @StateObject private var scroll=ScrollZoomMonitor()
    init(path:String) {
        let loaded=NSImage(contentsOfFile:path)
        self.path=path;image=loaded;pixels=loaded?.cgImage(forProposedRect:nil,context:nil,hints:nil)
    }
    init(image:NSImage) {
        self.path="";self.image=image;pixels=image.cgImage(forProposedRect:nil,context:nil,hints:nil)
    }
    static func fittedRect(image:CGSize,in size:CGSize)->CGRect {
        let scale=min(size.width/max(1,image.width),size.height/max(1,image.height))
        let fitted=CGSize(width:image.width*scale,height:image.height*scale)
        return CGRect(x:(size.width-fitted.width)/2,y:(size.height-fitted.height)/2,width:fitted.width,height:fitted.height)
    }
    var zoomLabel:String { !zoomOn ? "Off":level==0 ? "100%":"\(Int(level))×" }
    func stepLevel(_ step:Int) {
        guard zoomOn else { return }
        let index=HoverZoom.levels.firstIndex(of:level) ?? 2
        level=HoverZoom.levels[min(max(0,index+step),HoverZoom.levels.count-1)]
    }
    /// Draws the source pixels around the pointer, so 8× and 100% show real detail rather than an enlarged screen image.
    func loupe(_ cg:CGImage,rect:CGRect,point:CGPoint,diameter:CGFloat)->some View {
        let pixelsPerPoint=Double(cg.width)/Double(max(1,rect.width))
        let backing=Double(NSScreen.main?.backingScaleFactor ?? 2)
        let across=level==0 ? Double(diameter)*backing:Double(diameter)/level*pixelsPerPoint
        let scale=Double(diameter)/across
        let cx=Double(point.x-rect.minX)/Double(max(1,rect.width))*Double(cg.width)
        let cy=Double(point.y-rect.minY)/Double(max(1,rect.height))*Double(cg.height)
        let want=CGRect(x:cx-across/2,y:cy-across/2,width:across,height:across)
        let bounds=CGRect(x:0,y:0,width:cg.width,height:cg.height)
        let have=want.intersection(bounds)
        let crop=CGRect(x:floor(have.minX),y:floor(have.minY),width:ceil(have.width),height:ceil(have.height)).intersection(bounds)
        return ZStack(alignment:.topLeading) {
            Color.white
            if !have.isNull,crop.width>=1,crop.height>=1,let part=cg.cropping(to:crop) {
                Image(decorative:part,scale:1).resizable().interpolation(level==0 ? .none:.high)
                    .frame(width:crop.width*scale,height:crop.height*scale)
                    .offset(x:(crop.minX-want.minX)*scale,y:(crop.minY-want.minY)*scale)
            }
        }.frame(width:diameter,height:diameter).clipShape(Circle())
            .overlay(Circle().stroke(Color.white,lineWidth:3))
            .overlay(alignment:.bottom) { Text(zoomLabel).font(.system(size:10,weight:.semibold)).padding(.horizontal,7).padding(.vertical,3).background(Color.black.opacity(0.75)).clipShape(Capsule()).padding(.bottom,12) }
            .shadow(color:.black.opacity(0.3),radius:10)
    }
    var zoomMenu:some View {
        Menu {
            Picker("Zoom",selection:Binding(get:{zoomOn ? level:-1},set:{ value in
                if value<0 { zoomOn=false } else { level=value;zoomOn=true }
            })) {
                Text("Off").tag(-1.0)
                ForEach(HoverZoom.levels,id:\.self) { Text("\(Int($0))×").tag($0) }
                Text("Actual pixels").tag(0.0)
            }.pickerStyle(.inline).labelsHidden()
        } label: {
            Label(zoomLabel,systemImage:"plus.magnifyingglass").font(.system(size:11,weight:.medium))
                .padding(.horizontal,9).padding(.vertical,6).background(Color.black.opacity(0.7)).clipShape(Capsule())
        }.menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize().hoverFill(14,0.12).padding(10)
            .help("Hover zoom. Choose a level here, or scroll over the photo.")
    }
    var body:some View {
        GeometryReader { geometry in
            if let image=image,let pixels=pixels {
                let rect=Self.fittedRect(image:image.size,in:geometry.size)
                let diameter=min(260.0,min(geometry.size.width,geometry.size.height)*0.7)
                ZStack(alignment:.topLeading) {
                    Image(nsImage:image).resizable().frame(width:rect.width,height:rect.height).position(x:rect.midX,y:rect.midY)
                    if zoomOn,let point=pointer {
                        loupe(pixels,rect:rect,point:point,diameter:diameter)
                            .position(x:min(max(point.x,diameter/2),geometry.size.width-diameter/2),y:min(max(point.y,diameter/2),geometry.size.height-diameter/2))
                            .allowsHitTesting(false)
                    }
                }.frame(width:geometry.size.width,height:geometry.size.height).contentShape(Rectangle())
                    .onContinuousHover { phase in
                        switch phase {
                        case .active(let location):pointer=rect.contains(location) ? location:nil
                        case .ended:pointer=nil
                        }
                        scroll.hovering=pointer != nil && zoomOn
                    }
                    .overlay(alignment:.topTrailing) { zoomMenu }
            }
        }
        .onAppear { scroll.onStep={ step in stepLevel(step) };scroll.start() }
        .onDisappear { scroll.stop();scroll.hovering=false }
    }
}

/// A slider that follows the pointer locally and applies its value on release,
/// so dragging never triggers a preview render or a save on every step.
struct SliderRow:View {
    let title:String
    @Binding var value:Double
    let range:ClosedRange<Double>
    let text:(Double)->String
    let reset:Double
    let help:String
    @State private var draft:Double?
    @State private var editing=false
    var body:some View {
        HStack(spacing:8) {
            Text(title).font(.system(size:13)).frame(width:70,alignment:.leading)
                .help(help+"\nDouble-click the name to reset.").onTapGesture(count:2) { draft=nil;value=reset }
            Slider(value:Binding(get:{ draft ?? value },set:{ new in
                draft=new
                // Keyboard and accessibility steps have no drag to finish, so they apply at once.
                if !editing { value=new;draft=nil }
            }),in:range) { dragging in
                editing=dragging
                if !dragging,let final=draft { value=final;draft=nil }
            }.controlSize(.small).tint(accent)
            Text(text(draft ?? value)).font(.system(size:11,design:.monospaced)).foregroundStyle(draft == nil ? muted:Color.white).frame(width:44,alignment:.trailing)
        }.frame(height:26)
    }
}

/// Steps through a folder's photos (turntable angles) without leaving the folder view.
struct AngleBar:View {
    @ObservedObject var s:Studio
    let job:BatchJob
    @State private var drag:Double?
    var body:some View {
        let count=job.sources.count
        let index=s.angleIndex(job)
        let shown=min(max(0,Int((drag ?? Double(index)).rounded())),count-1)
        let known=job.sources.contains { Studio.degrees($0) != nil }
        let current=Studio.degrees(job.sources[index])
        HStack(spacing:10) {
            Button { s.stepAngle(-1) } label: { Image(systemName:"chevron.left").frame(width:26,height:24) }.buttonStyle(ToolButtonStyle())
                .keyboardShortcut(.leftArrow,modifiers:[]).help("Previous photo (←)")
            Button { s.stepAngle(1) } label: { Image(systemName:"chevron.right").frame(width:26,height:24) }.buttonStyle(ToolButtonStyle())
                .keyboardShortcut(.rightArrow,modifiers:[]).help("Next photo (→)")
            Text(s.angleLabel(job,shown)).font(.system(size:13,weight:.semibold,design:.monospaced)).frame(width:56,alignment:.leading)
            Slider(value:Binding(get:{ drag ?? Double(index) },set:{ drag=$0 }),in:0...Double(max(1,count-1)),step:1) { editing in
                if !editing,let value=drag { s.setAngle(job.id,Int(value.rounded()));drag=nil }
            }.controlSize(.small).tint(accent)
            Text("\(shown+1)/\(count)").font(.system(size:11,design:.monospaced)).foregroundStyle(muted)
            if known {
                HStack(spacing:4) {
                    ForEach([0,90,180,270],id:\.self) { degrees in
                        Button { s.jumpAngle(degrees) } label: {
                            Text("\(degrees)°").font(.system(size:11)).padding(.horizontal,7).padding(.vertical,3)
                                .background(current==degrees ? accent.opacity(0.35):Color.white.opacity(0.07)).clipShape(Capsule())
                        }.buttonStyle(ToolButtonStyle(radius:10))
                    }
                }
            }
        }.buttonStyle(.plain).disabled(s.queueRunning || s.running).frame(height:34)
    }
}

struct StudioView:View {
    @StateObject var s=Studio()
    @State var confirmStop=false
    @AppStorage("panelTone") var openTone=true
    @AppStorage("panelDetail") var openDetail=true
    @AppStorage("panelShadow") var openShadow=true
    @AppStorage("panelOutput") var openOutputGroup=true
    @AppStorage("panelCutout") var openCutout=false

    var body:some View {
        VStack(spacing:0) {
            toolbar
            Divider()
            if s.queueVisible { queueLayout }
            else if s.inputs.isEmpty { welcome }
            else { photoLayout }
            if showFooter { Divider();footer }
        }
        .background(appBackground).foregroundStyle(Color.white.opacity(0.92)).preferredColorScheme(.dark)
        .frame(minWidth:1024,minHeight:680)
        .onDrop(of:[UTType.fileURL.identifier],isTargeted:nil,perform:s.receiveFolders)
        .onAppear { if s.queueVisible { s.openQueue() } }
        .onReceive(NotificationCenter.default.publisher(for:NSApplication.willTerminateNotification)) { _ in s.shutdown() }
        .sheet(isPresented:$s.completedVisible) { completedFolders }
        .alert("Photo processing",isPresented:Binding(get:{s.error != nil},set:{if !$0{s.error=nil}})) { Button("OK"){s.error=nil} } message:{Text(s.error ?? "")}
    }

    // MARK: toolbar
    var toolbar:some View {
        HStack(spacing:10) {
            if s.queueVisible { queueToolbar } else { photoToolbar }
        }.padding(.horizontal,16).frame(height:46)
    }
    func iconButton(_ symbol:String,_ help:String,action:@escaping ()->Void)->some View {
        Button(action:action) { Image(systemName:symbol).font(.system(size:14)).frame(width:28,height:26) }.buttonStyle(ToolButtonStyle()).help(help)
    }
    @ViewBuilder var queueToolbar:some View {
        Button(action:s.chooseBatches) {
            Label("Add folders",systemImage:"plus").font(.system(size:13,weight:.medium)).padding(.horizontal,11).padding(.vertical,6)
        }.buttonStyle(AccentButtonStyle()).keyboardShortcut("o",modifiers:.command).disabled(s.queueRunning)
        iconButton("arrow.clockwise","Refresh folders: pick up photos added to or removed from queued folders, keeping their recipes.") { s.refreshBatchFolders() }
            .disabled(s.queueRunning || s.activeBatches.isEmpty)
        Spacer()
        if s.queue.jobs.contains(where:{$0.status=="error"}) { Button("Retry failed",action:s.retryFailedBatches).disabled(s.queueRunning) }
        folderLayoutMenu
        iconButton("arrow.up.right.square","Open exports in Finder",action:s.openOutput)
    }
    var folderLayoutMenu:some View {
        Menu {
            Picker("Folders",selection:Binding(get:{s.foldersHidden ? 2:(s.foldersOnLeft ? 0:1)},set:{ value in
                s.foldersHidden=value==2
                if value<2 { s.foldersOnLeft=value==0 }
            })) {
                Text("Folders on the left").tag(0)
                Text("Folders below").tag(1)
                Text("Hide folders").tag(2)
            }.pickerStyle(.inline).labelsHidden()
            Button("Show or hide folders") { s.foldersHidden.toggle() }.keyboardShortcut("s",modifiers:[.command,.control])
        } label: { Image(systemName:"sidebar.left").font(.system(size:14)).frame(width:28,height:26) }
            .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize().hoverFill().help("Folder list position (⌃⌘S hides it)")
    }
    var photoTitle:String {
        let job=s.previewFolderID.flatMap { id in s.queue.jobs.first { $0.id==id } }
        let name=job?.shortName ?? "Photos"
        guard let path=s.selectedPath else { return name }
        return name+" · "+(Studio.degrees(path).map { String(format:"%03d°",$0) } ?? "Photo \(s.selected+1)")
    }
    @ViewBuilder var photoToolbar:some View {
        Button { s.openQueue() } label: { Label("Folders",systemImage:"chevron.left").font(.system(size:13,weight:.medium)).padding(.horizontal,8).padding(.vertical,5) }
            .buttonStyle(ToolButtonStyle()).disabled(s.running || s.queueRunning)
        Text(photoTitle).font(.system(size:13,weight:.semibold)).lineLimit(1)
        Spacer()
        if s.exportFraming=="both" {
            Picker("Crop",selection:$s.previewFraming) { Text("Centred").tag("centered");Text("Fixed").tag("fixed") }
                .pickerStyle(.segmented).labelsHidden().frame(width:150).help("Which crop to preview")
        }
        Picker("Preview",selection:$s.mode) { Text("Edited").tag("Edited");Text("Original").tag("Original");Text("Compare").tag("Compare") }
            .pickerStyle(.segmented).labelsHidden().frame(width:210)
        iconButton("arrow.up.right.square","Open exports in Finder",action:s.openOutput)
    }

    // MARK: folder screen
    var queueLayout:some View {
        HStack(spacing:0) {
            if s.foldersOnLeft && !s.foldersHidden && !s.activeBatches.isEmpty { folderSidebar;Divider() }
            queueCenter
            Divider()
            inspector.frame(width:316)
        }
    }
    var queueCenter:some View {
        VStack(spacing:0) {
            if let job=s.selectedBatch {
                folderHeader(job)
                stage(job).padding(.horizontal,16)
                if job.sources.count>1 { AngleBar(s:s,job:job).padding(.horizontal,16) }
                if !s.foldersOnLeft && !s.foldersHidden { Divider().padding(.top,6);folderStrip } else { Spacer().frame(height:10) }
            } else { emptyQueue }
        }.frame(maxWidth:.infinity,maxHeight:.infinity)
    }
    var emptyQueue:some View {
        VStack(spacing:14) {
            Image(systemName:"folder.badge.plus").font(.system(size:38,weight:.light)).foregroundStyle(muted)
            Text(s.completedBatches.isEmpty ? "Drop product folders here":"No folders left in the queue").font(.system(size:15)).foregroundStyle(muted)
            Button("Add folders",action:s.chooseBatches).disabled(s.queueRunning)
            if !s.completedBatches.isEmpty { Button { s.completedVisible=true } label: { Text("Completed (\(s.completedBatches.count))").padding(.horizontal,8).padding(.vertical,4) }.buttonStyle(ToolButtonStyle()) }
        }.frame(maxWidth:.infinity,maxHeight:.infinity)
    }
    func folderHeader(_ job:BatchJob)->some View {
        let recipeName=job.started_recipe != nil ? "Saved recipe":job.override == nil ? "Shared recipe":"Folder recipe"
        var detail="\(job.sources.count) photos · \(recipeName)"
        if !["queued","complete"].contains(job.status) { detail+=" · "+job.status.capitalized }
        return HStack(spacing:10) {
            VStack(alignment:.leading,spacing:2) {
                Text(job.name).font(.system(size:15,weight:.semibold)).lineLimit(1).truncationMode(.middle)
                Text(job.message ?? detail).font(.system(size:12)).foregroundStyle(job.message != nil ? Color.yellow:muted).lineLimit(1).help(job.message ?? detail)
            }
            Spacer()
            if s.exportFraming=="both" {
                Picker("Crop",selection:$s.previewFraming) { Text("Centred").tag("centered");Text("Fixed").tag("fixed") }
                    .pickerStyle(.segmented).labelsHidden().frame(width:150).help("Which crop to preview")
            }
            Button { s.previewBatch(job) } label: { Label("All photos",systemImage:"square.grid.2x2").font(.system(size:12)).padding(.horizontal,9).padding(.vertical,5).background(RoundedRectangle(cornerRadius:6).fill(Color.white.opacity(0.07))) }.buttonStyle(ToolButtonStyle())
                .help("Preview and edit every photo in this folder").disabled(s.queueRunning)
            folderMenu(job)
        }.padding(.horizontal,16).frame(height:50)
    }
    func folderMenu(_ job:BatchJob)->some View {
        Menu {
            let targets=s.actionTargets(job.id),many=targets.count>1
            Button(many ? "Refresh photos in \(targets.count) folders":"Refresh photos") { s.refreshBatchFolders(targets) }
            if job.override != nil { Button(many ? "Use shared recipe for \(targets.count) folders":"Use shared recipe") { s.useShared(job.id) }.disabled(job.started_recipe != nil) }
            if job.output != nil { Button("Open images") { s.openExport(job.id) } }
            Divider()
            Button(many ? "Mark \(targets.count) folders completed":"Mark completed") { s.archiveBatches(targets) }
            Button(many ? "Remove \(targets.count) folders":"Remove folder") { s.removeBatches(targets) }
        } label: { Image(systemName:"ellipsis.circle").font(.system(size:14)) }
            .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize().frame(width:26,height:24).hoverFill().disabled(s.queueRunning)
    }
    func stage(_ job:BatchJob)->some View {
        ZStack {
            stageBackground
            if let image=s.batchDetailImage,s.batchDetailJobID==job.id {
                ZoomPhotoView(image:image).id(s.batchDetailSignature).padding(10)
            } else if let image=s.batchImages[job.id] {
                Image(nsImage:image).resizable().aspectRatio(contentMode:.fit).padding(10)
            } else {
                VStack(spacing:10) {
                    Image(systemName:"camera.aperture").font(.system(size:34,weight:.light)).foregroundStyle(accent)
                    Text("Preparing preview").font(.system(size:13)).foregroundStyle(muted)
                }
            }
        }
        .overlay(alignment:.bottomLeading) { previewStatus(job).padding(10) }
        .clipShape(RoundedRectangle(cornerRadius:8))
        .frame(maxWidth:.infinity,minHeight:200,maxHeight:.infinity)
    }
    func statusPill<C:View>(@ViewBuilder _ content:()->C)->some View {
        HStack(spacing:6) { content() }.font(.system(size:12)).padding(.horizontal,10).padding(.vertical,6)
            .background(Color.black.opacity(0.65)).clipShape(Capsule())
    }
    @ViewBuilder func previewStatus(_ job:BatchJob)->some View {
        let note=s.batchPreviewNotes[job.id] ?? ""
        if s.batchPreviewActive?.id==job.id || s.needsBatchPreview(job) {
            statusPill { ProgressView().controlSize(.small);Text(note.hasSuffix("…") ? note:"Updating preview…").lineLimit(1) }
        } else if s.batchPreviewFailures[job.id] != nil {
            statusPill {
                Image(systemName:"exclamationmark.triangle.fill").foregroundStyle(.yellow)
                Text(note).lineLimit(1)
                Button("Retry") { s.retryBatchPreview(job.id) }.buttonStyle(ToolButtonStyle()).foregroundStyle(accent)
            }
        } else if note.hasPrefix("Review:") {
            statusPill { Image(systemName:"exclamationmark.triangle.fill").foregroundStyle(.yellow);Text(note).lineLimit(2) }.frame(maxWidth:460,alignment:.leading)
        } else if !note.isEmpty {
            Image(systemName:"info.circle").foregroundStyle(muted).padding(6).help(note)
        }
    }

    // MARK: folder list
    func folderListHeader(compact:Bool)->some View {
        HStack(spacing:8) {
            Toggle("Select all folders",isOn:Binding(get:{ !s.activeBatches.isEmpty && s.exportableBatches.count==s.activeBatches.count },set:{ $0 ? s.selectAllBatches():s.clearBatchSelection() }))
                .toggleStyle(.checkbox).labelsHidden().disabled(s.queueRunning).help("Select all or none")
            Text("\(s.selectedBatchCount)/\(s.activeBatches.count)").font(.system(size:12,design:.monospaced)).foregroundStyle(muted)
                .help("Folders selected for export. Click a folder to preview it. Shift-click selects a range, Command-click adds or removes one, and dragging reorders. Folders are processed in this order.")
            Spacer()
            if !compact && !s.completedBatches.isEmpty {
                Button { s.completedVisible=true } label: { Text("Completed \(s.completedBatches.count)").font(.system(size:12)).padding(.horizontal,7).padding(.vertical,3) }.buttonStyle(ToolButtonStyle())
            }
            Menu {
                if compact && !s.completedBatches.isEmpty { Button("Completed folders (\(s.completedBatches.count))…") { s.completedVisible=true };Divider() }
                Button("Mark \(s.selectedBatchCount) selected completed") { s.archiveBatches(s.exportableBatches.map(\.id)) }.disabled(s.exportableBatches.isEmpty)
            } label: { Image(systemName:"ellipsis").frame(width:22,height:20) }.menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize().hoverFill().disabled(s.queueRunning)
        }
    }
    func inRun(_ folder:BatchJob)->Bool { folder.include != false }
    @ViewBuilder func statusIcon(_ folder:BatchJob)->some View {
        if s.queueRunning && inRun(folder) && ["queued","paused","retrying"].contains(folder.status) {
            Image(systemName:"circle.dotted").font(.system(size:12)).foregroundStyle(muted).help("Waiting in this run")
        } else {
            statusSymbol(folder)
        }
    }
    @ViewBuilder func statusSymbol(_ folder:BatchJob)->some View {
        switch folder.status {
        case "running","retrying": ProgressView().controlSize(.mini)
        case "complete": Image(systemName:"checkmark.circle.fill").font(.system(size:12)).foregroundStyle(.green)
        case "error": Image(systemName:"exclamationmark.triangle.fill").font(.system(size:12)).foregroundStyle(.yellow)
        case "paused": Image(systemName:"pause.circle").font(.system(size:12)).foregroundStyle(muted)
        case "stopped": Image(systemName:"stop.circle").font(.system(size:12)).foregroundStyle(muted)
        default: EmptyView()
        }
    }
    func folderTile(_ folder:BatchJob,thumb:CGSize)->some View {
        let selected=s.selectedBatchID==folder.id
        let checked=folder.include != false
        return ZStack(alignment:.topLeading) {
            VStack(alignment:.leading,spacing:5) {
                ZStack {
                    Color.white
                    if let image=s.batchImages[folder.id] { Image(nsImage:image).resizable().aspectRatio(contentMode:.fit) }
                    else { Image(systemName:"photo").foregroundStyle(.gray) }
                }.frame(width:thumb.width,height:thumb.height).clipShape(RoundedRectangle(cornerRadius:5))
                    .overlay(alignment:.bottom) {
                        if folder.status=="running" || (folder.status=="paused" && folder.completed>0) {
                            let fraction=Double(s.finishedPhotos(folder))/Double(max(1,folder.sources.count))
                            GeometryReader { geo in
                                ZStack(alignment:.leading) {
                                    Capsule().fill(Color.black.opacity(0.18))
                                    Capsule().fill(folder.status=="running" ? accent:muted).frame(width:geo.size.width*fraction)
                                }
                            }.frame(height:4).padding(.horizontal,6).padding(.bottom,5)
                        }
                    }
                HStack(spacing:5) {
                    Text(folder.shortName).font(.system(size:12,weight:.medium)).lineLimit(1)
                    Spacer(minLength:0)
                    statusIcon(folder)
                }.frame(width:thumb.width)
            }.padding(6)
                .hoverFill(8,0.07)
                .background(RoundedRectangle(cornerRadius:8).fill(selected ? accent.opacity(0.3):checked ? accent.opacity(0.16):Color.white.opacity(0.05)))
                .clipShape(RoundedRectangle(cornerRadius:8))
                .overlay(RoundedRectangle(cornerRadius:8).stroke(s.batchDropTarget==folder.id ? Color.white:selected ? accent:Color.clear,lineWidth:2))
                .opacity(s.queueRunning && !inRun(folder) ? 0.4:1)
                .contentShape(RoundedRectangle(cornerRadius:8))
                .onTapGesture { s.clickBatch(folder.id,shift:NSEvent.modifierFlags.contains(.shift),command:NSEvent.modifierFlags.contains(.command)) }
                .accessibilityAddTraits(.isButton)
            Toggle("Export \(folder.name)",isOn:Binding(get:{folder.include != false},set:{s.setBatchIncluded(folder.id,$0)}))
                .toggleStyle(.checkbox).labelsHidden().help("Include this folder in the export")
                .padding(3).background(Color.black.opacity(0.6)).clipShape(RoundedRectangle(cornerRadius:5))
                .padding(10).disabled(s.queueRunning)
        }
        .onDrag { s.batchDragProvider(folder.id) }
        .onDrop(of:[Studio.batchDragType],delegate:BatchOrderDrop(studio:s,target:folder.id))
        .contextMenu {
            let targets=s.actionTargets(folder.id),many=targets.count>1
            Button(many ? "Mark \(targets.count) folders completed":"Mark completed") { s.archiveBatches(targets) }.disabled(s.queueRunning)
            Button(many ? "Remove \(targets.count) folders":"Remove folder") { s.removeBatches(targets) }.disabled(s.queueRunning)
        }
        .help("\(folder.name)\n\(folder.status=="running" || folder.status=="paused" ? "\(s.finishedPhotos(folder))/\(folder.sources.count) photos exported":"\(folder.sources.count) photos") · \(folder.status)")
    }
    var folderStrip:some View {
        VStack(alignment:.leading,spacing:6) {
            folderListHeader(compact:false).padding(.horizontal,16)
            ScrollViewReader { reader in
                ScrollView(.horizontal) {
                    LazyHStack(spacing:8) {
                        ForEach(s.activeBatches) { folder in folderTile(folder,thumb:CGSize(width:104,height:64)).id(folder.id) }
                        RoundedRectangle(cornerRadius:6).fill(s.batchDropTarget=="end" ? accent.opacity(0.4):Color.white.opacity(0.04))
                            .frame(width:30,height:104).overlay(Image(systemName:"arrow.right.to.line").foregroundStyle(muted))
                            .onDrop(of:[Studio.batchDragType],delegate:BatchOrderDrop(studio:s,target:nil))
                    }.padding(.horizontal,16).padding(.vertical,2).animation(.easeInOut(duration:0.22),value:s.activeBatches.map(\.id))
                }.frame(height:114)
                    .onChange(of:s.selectedBatchID) { _,id in if let id=id { withAnimation { reader.scrollTo(id,anchor:.center) } } }
                    .onChange(of:s.queue.active_job_id) { _,id in if s.queueRunning,let id=id { withAnimation { reader.scrollTo(id,anchor:.center) } } }
            }
        }.padding(.vertical,8)
    }
    var folderSidebar:some View {
        VStack(alignment:.leading,spacing:8) {
            folderListHeader(compact:true).padding(.horizontal,12)
            ScrollViewReader { reader in
                ScrollView {
                    LazyVStack(spacing:8) {
                        ForEach(s.activeBatches) { folder in folderTile(folder,thumb:CGSize(width:164,height:88)).id(folder.id) }
                        RoundedRectangle(cornerRadius:6).fill(s.batchDropTarget=="end" ? accent.opacity(0.4):Color.white.opacity(0.04))
                            .frame(height:28).overlay(Image(systemName:"arrow.down.to.line").foregroundStyle(muted))
                            .onDrop(of:[Studio.batchDragType],delegate:BatchOrderDrop(studio:s,target:nil))
                    }.padding(.horizontal,10).padding(.vertical,2).animation(.easeInOut(duration:0.22),value:s.activeBatches.map(\.id))
                }.onChange(of:s.selectedBatchID) { _,id in if let id=id { withAnimation { reader.scrollTo(id,anchor:.center) } } }
                    .onChange(of:s.queue.active_job_id) { _,id in if s.queueRunning,let id=id { withAnimation { reader.scrollTo(id,anchor:.center) } } }
            }
        }.padding(.vertical,10).frame(width:196)
    }

    var completedFolders:some View {
        VStack(alignment:.leading,spacing:18) {
            HStack { Text("Completed folders").font(.title2).fontWeight(.semibold);Spacer();Button("Done") { s.completedVisible=false } }
            if s.completedBatches.isEmpty { Text("Nothing marked completed yet.").foregroundStyle(muted).frame(maxWidth:.infinity,minHeight:140) }
            else {
                ScrollView {
                    LazyVStack(spacing:10) {
                        ForEach(s.completedBatches) { job in
                            HStack(spacing:14) {
                                ZStack {
                                    Color.white
                                    if let image=s.batchImages[job.id] { Image(nsImage:image).resizable().aspectRatio(contentMode:.fit) }
                                    else { Image(systemName:"checkmark.circle").foregroundStyle(.gray) }
                                }.frame(width:86,height:64).clipShape(RoundedRectangle(cornerRadius:6))
                                VStack(alignment:.leading,spacing:5) {
                                    Text(job.name).fontWeight(.medium).lineLimit(1)
                                    Text("\(job.sources.count) photos · \(job.status.capitalized)").font(.caption).foregroundStyle(muted)
                                }
                                Spacer()
                                if job.output != nil { Button("Open images") { s.openExport(job.id) } }
                                Button("Restore to queue") { s.restoreBatch(job.id) }.disabled(s.queueRunning || s.running)
                            }.padding(10).background(Color.white.opacity(0.05)).clipShape(RoundedRectangle(cornerRadius:8))
                        }
                    }
                }.frame(minHeight:180,maxHeight:440)
            }
        }.padding(24).frame(width:720).preferredColorScheme(.dark)
    }

    var welcome:some View {
        VStack(spacing:16) {
            Image(systemName:"folder.badge.plus").font(.system(size:44,weight:.light)).foregroundStyle(muted)
            Text("Add a product folder to begin").font(.system(size:17))
            Button("Add folders",action:s.chooseBatches).keyboardShortcut("o",modifiers:.command)
        }.frame(maxWidth:.infinity,maxHeight:.infinity)
    }

    // MARK: photo view
    var photoLayout:some View {
        HStack(spacing:0) { library.frame(width:210);Divider();workspace;Divider();inspector.frame(width:316) }
    }
    var library:some View {
        VStack(alignment:.leading,spacing:8) {
            HStack(spacing:8) {
                Toggle("Select all photos",isOn:Binding(get:{ !s.inputs.isEmpty && s.checked.count==s.inputs.count },set:{ $0 ? s.selectAll():s.deselectAll() }))
                    .toggleStyle(.checkbox).labelsHidden().disabled(s.running).help("Select all or none")
                Text("\(s.checked.count)/\(s.inputs.count)").font(.system(size:12,design:.monospaced)).foregroundStyle(muted).help("Checked photos receive edits and are exported. Shift-click checks a range from the last clicked photo; Command-click checks or unchecks one.")
                Spacer()
            }.padding(.horizontal,14).padding(.top,10)
            ScrollView {
                LazyVStack(spacing:2) {
                    ForEach(Array(s.inputs.enumerated()),id:\.element) { index,url in photoRow(index,url) }
                }.padding(.horizontal,8)
            }
        }
    }
    func photoRow(_ index:Int,_ url:URL)->some View {
        let photo=s.photos.first(where:{$0.source==url.path})
        let packaging=s.packagingRecipeFor(url.path) != nil
        return HStack(spacing:8) {
            Toggle("Include photo \(index+1)",isOn:Binding(get:{s.included.contains(url.path)},set:{value in if value{s.included.insert(url.path)}else{s.included.remove(url.path)}})).toggleStyle(.checkbox).labelsHidden().disabled(s.running)
            Button { s.clickPhoto(index,shift:NSEvent.modifierFlags.contains(.shift),command:NSEvent.modifierFlags.contains(.command)) } label: {
                HStack(spacing:8) {
                    if let photo=photo { PhotoView(path:photo.jpeg).frame(width:28,height:28).background(Color.white).clipShape(RoundedRectangle(cornerRadius:4)) }
                    Text(Studio.degrees(url.path).map { String(format:"%03d°",$0) } ?? "Photo \(String(format:"%02d",index+1))").font(.system(size:13,weight:.medium,design:.monospaced))
                    if packaging { Image(systemName:"shippingbox").font(.system(size:11)).foregroundStyle(muted).help("Packaging photo") }
                    Spacer(minLength:0)
                    if s.running && s.activeName==url.lastPathComponent {
                        ProgressView(value:s.photoProgress).tint(accent).frame(width:44)
                    }
                    if !(photo?.review ?? s.previewReviews[url.path] ?? []).isEmpty {
                        Image(systemName:"exclamationmark.triangle.fill").foregroundStyle(.yellow).help("Compare separate pieces with the original")
                    }
                }.contentShape(Rectangle())
            }.buttonStyle(.plain)
        }.padding(.horizontal,8).frame(height:34).hoverFill(6,0.06).background(s.selected==index ? accent.opacity(0.24):Color.clear).clipShape(RoundedRectangle(cornerRadius:6))
            .contextMenu {
                Button(packaging ? "Unmark packaging photo":"Mark as packaging photo") { s.markPackaging(url.path,!packaging) }
            }
    }
    var workspace:some View {
        VStack(spacing:0) {
            photoStage.padding(12)
            if s.running { processingPanel.padding(.horizontal,12).padding(.bottom,12) }
        }.frame(maxWidth:.infinity,maxHeight:.infinity)
    }
    var photoStage:some View {
        ZStack {
            stageBackground
            if s.mode=="Compare",let original=s.currentOriginal,let edited=s.currentEdited {
                HStack(spacing:10) { previewTile(original,"Original");previewTile(edited,"Edited") }.padding(12)
            } else if let path=(s.mode=="Original" ? s.currentOriginal:(s.currentEdited ?? s.currentOriginal)) {
                ZoomPhotoView(path:path).id(path).padding(10)
            } else {
                VStack(spacing:12) {
                    Image(systemName:"camera.aperture").font(.system(size:38,weight:.light)).foregroundStyle(accent)
                    Text(s.previewLoading ? "Preparing this photo":"Select a photo to preview").font(.system(size:15)).foregroundStyle(muted)
                    if s.previewLoading { ProgressView().controlSize(.small) }
                }
            }
        }
        .overlay(alignment:.bottomLeading) { photoStatus.padding(10) }
        .clipShape(RoundedRectangle(cornerRadius:8))
        .frame(maxWidth:.infinity,maxHeight:.infinity)
    }
    @ViewBuilder var photoStatus:some View {
        if s.previewLoading && !s.running {
            statusPill { ProgressView().controlSize(.small);Text("Updating preview…") }
        } else if !s.currentReview.isEmpty {
            statusPill { Image(systemName:"exclamationmark.triangle.fill").foregroundStyle(.yellow);Text(s.currentReview.joined(separator:". ")).lineLimit(2) }.frame(maxWidth:460,alignment:.leading)
        } else {
            Image(systemName:s.matchesExport ? "checkmark.circle":"sparkles").foregroundStyle(muted).padding(6)
                .help(s.running ? "Showing selected photo":s.matchesExport ? "This is the exported image. \(s.previewNote)":"Live recipe preview. \(s.previewNote)")
        }
    }
    func previewTile(_ path:String,_ label:String)->some View {
        VStack(spacing:6) {
            ZoomPhotoView(path:path).id(path).frame(maxWidth:.infinity,maxHeight:.infinity).background(Color.white)
            Text(label).font(.system(size:11)).foregroundStyle(muted)
        }
    }
    var processingPanel:some View {
        VStack(alignment:.leading,spacing:8) {
            HStack {
                Text(s.finishing ? "Finishing exports":"Photo \(s.activePhoto) of \(s.batchTotal)").font(.system(size:13,weight:.semibold))
                Text(s.phase).font(.system(size:12)).foregroundStyle(muted)
                Spacer()
                Text(Studio.duration(s.photoElapsed)).font(.system(size:14,weight:.medium,design:.monospaced))
            }
            ProgressView(value:s.photoProgress).tint(accent)
        }.padding(12).background(Color.white.opacity(0.05)).clipShape(RoundedRectangle(cornerRadius:8))
    }

    // MARK: settings panel
    var inspectorTitle:String {
        if s.editingPackagePath != nil { return "Packaging photo" }
        if s.queueVisible {
            guard let id=s.recipeTarget else { return "Shared recipe" }
            let n=s.actionTargets(id).count
            return n>1 ? "\(n) folders":"Folder recipe"
        }
        return s.checked.count==1 ? "1 photo":"\(s.checked.count) photos"
    }
    var inspectorHelp:String {
        if s.editingPackagePath != nil { return "Packaging settings override the folder recipe for this photo." }
        if s.queueVisible {
            if s.recipeTarget == nil { return "Editing the shared recipe applies to every folder that uses it. Select a folder to edit that folder instead." }
            return "Sliders show the settings of the photo in the preview. Each edit applies that control to every photo in the selected folders; other individual suggestions stay as they are."
        }
        return "Sliders show the photo you are viewing. Edits apply to the \(s.checked.count) checked photos." + (s.checked.count==s.inputs.count ? " With every photo checked, edits also update the folder settings.":" Unchecked photos keep their settings.")
    }
    var inspectorHeader:some View {
        HStack(spacing:8) {
            Text(inspectorTitle).font(.system(size:13,weight:.semibold)).lineLimit(1).help(inspectorHelp)
            if s.editingAdjustmentPath != nil,let a=s.adjustmentFor(s.editingAdjustmentPath ?? "") {
                Image(systemName:"sparkles").font(.system(size:11)).foregroundStyle(accent)
                    .help(a.manual_fields.isEmpty ? (a.confidence=="matched" ? "Your last settings for this photo":"Suggested from your processing history") : "Your corrections are saved when exported")
            }
            Spacer()
            if s.learningBusy {
                ProgressView().controlSize(.small)
                Button(action:s.cancelLearning) { Text("Cancel").font(.system(size:12)).padding(.horizontal,6).padding(.vertical,3) }.buttonStyle(ToolButtonStyle())
            } else {
                iconButton("wand.and.stars","Suggest settings from your processing history") {
                    s.requestLearning(s.queueVisible ? s.actionTargets(s.selectedBatchID ?? "").flatMap { id in s.queue.jobs.first{$0.id==id}?.sources ?? [] } : s.checked.map(\.path))
                }.disabled(s.running || s.queueRunning)
            }
            iconButton("arrow.counterclockwise","Reset every setting to its default",action:s.resetDefaults).disabled(s.running || s.queueRunning || s.recipeLocked)
            Menu {
                Toggle("Learn settings for new photos",isOn:$s.autoLearn)
                if !s.queueVisible,let path=s.selectedPath {
                    Toggle("Packaging photo",isOn:Binding(get:{s.packagingRecipeFor(path) != nil},set:{s.markPackaging(path,$0)}))
                }
                if s.editingPackagePath != nil {
                    Button("Use packaging default",action:s.usePackagingDefault)
                    Button("Save as packaging default",action:s.savePackagingDefault)
                }
            } label: { Image(systemName:"ellipsis.circle").font(.system(size:14)).frame(width:24,height:26) }
                .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize().hoverFill()
        }
    }
    var inspector:some View {
        VStack(spacing:0) {
            inspectorHeader.padding(.horizontal,16).frame(height:44)
            Divider()
            ScrollView {
                VStack(alignment:.leading,spacing:4) {
                    if s.learningBusy || s.learningNoteFresh { Text(s.learningNote).font(.system(size:11)).foregroundStyle(muted).padding(.bottom,4) }
                    toneGroup;Divider().padding(.vertical,2)
                    detailGroup;Divider().padding(.vertical,2)
                    shadowGroup;Divider().padding(.vertical,2)
                    outputGroup;Divider().padding(.vertical,2)
                    cutoutGroup
                }.padding(16)
            }.disabled(s.running || s.queueRunning || s.recipeLocked)
            Divider()
            processArea
        }.background(panelBackground)
    }
    func panelSection<C:View>(_ title:String,_ open:Binding<Bool>,@ViewBuilder _ content:()->C)->some View {
        VStack(alignment:.leading,spacing:4) {
            Button { withAnimation(.easeOut(duration:0.15)) { open.wrappedValue.toggle() } } label: {
                HStack(spacing:6) {
                    Image(systemName:"chevron.right").font(.system(size:10,weight:.bold)).rotationEffect(.degrees(open.wrappedValue ? 90:0)).frame(width:10)
                    Text(title).font(.system(size:13,weight:.semibold))
                    Spacer()
                }.padding(.vertical,5).padding(.horizontal,4).contentShape(Rectangle())
            }.buttonStyle(ToolButtonStyle()).padding(.horizontal,-4)
            if open.wrappedValue { VStack(alignment:.leading,spacing:4) { content() }.padding(.leading,16).padding(.bottom,4) }
        }
    }
    func control(_ title:String,_ value:Binding<Double>,_ range:ClosedRange<Double>,_ text:@escaping (Double)->String,reset:Double,help:String)->some View {
        SliderRow(title:title,value:value,range:range,text:text,reset:reset,help:help)
    }
    func check(_ title:String,_ value:Binding<Bool>,help:String)->some View {
        Toggle(title,isOn:value).toggleStyle(.checkbox).font(.system(size:13)).help(help).frame(height:24)
    }
    func pct(_ v:Double)->String { Self.pct(v) }
    /// "+12", "−3" or "0": a value that rounds to zero never shows as "-0".
    static func signed(_ v:Double,_ digits:Int)->String {
        let text=String(format:"%+.\(digits)f",v)
        return Double(text)==0 ? String(format:"%.\(digits)f",0.0):text
    }
    static func pct(_ v:Double)->String { String(format:"%.0f%%",v*100) }
    var toneGroup:some View {
        panelSection("Tone",$openTone) {
            control("Exposure",$s.exposure,-0.5...1.5,{ Self.signed($0,2) },reset:0.65,help:"Brightness in stops. Highlights are compressed to protect pale bricks.")
            control("Contrast",$s.contrast,-1...1,{ Self.signed($0*100,0) },reset:0,help:"Gentle contrast curve.")
            control("Whites",$s.whites,-1...1,{ Self.signed($0*100,0) },reset:0,help:"Below zero brings overexposed white bricks down; mid-tones and shadows stay put. Above zero lifts only the brightest tones.")
            control("Warmth",$s.warmth,-1...1,{ Self.signed($0*100,0) },reset:0.25,help:"Colour temperature on top of the automatic neutral correction.")
        }
    }
    var detailGroup:some View {
        panelSection("Detail",$openDetail) {
            control("Sharpness",$s.sharpness,0...2,{ pct($0) },reset:0.75,help:"Output sharpening.")
            control("Noise",$s.denoise,0...2,{ pct($0) },reset:0,help:"Edge-preserving noise reduction.")
        }
    }
    var shadowGroup:some View {
        panelSection("Shadow",$openShadow) {
            control("Strength",$s.shadow,0...2,{ $0==0 ? "Off":pct($0) },reset:2,help:"Contact shadow strength recovered from the photographed floor. 0 is off.")
            check("Enhanced shadows",$s.enhancedShadow,help:"Recover wider photographed floor shadows. Off uses the original shadow method.")
            check("Remove floor patches",$s.softShadowDetection,help:"Rechecks grey floor caught in the cutout, including solid patches beside wheels. Enable if a hard floor patch is attached to the model or floating beneath it, and inspect pale or grey parts near the base.")
        }
    }
    var outputGroup:some View {
        panelSection("Output",$openOutputGroup) {
            HStack(spacing:8) {
                Picker("Canvas",selection:$s.aspect) { Text("Square 1:1").tag("square");Text("Landscape 3:2").tag("landscape");Text("Portrait 4:5").tag("portrait");Text("Wide 16:9").tag("wide") }.labelsHidden()
                Picker("Long edge",selection:$s.size) { Text("1600 px").tag(1600);Text("2400 px").tag(2400);Text("3200 px").tag(3200) }.labelsHidden().frame(width:96)
            }.help("Canvas shape and long edge")
            Picker("Framing",selection:$s.exportFraming) { Text("Centred").tag("centered");Text("Fixed").tag("fixed");Text("Both").tag("both") }
                .pickerStyle(.segmented).labelsHidden()
                .help("Centred keeps every view centred at one shared scale. Fixed frame keeps one crop and scale for the whole rotation and preserves original positions. Both exports the two.")
            control("Fill",$s.fill,0.60...0.92,{ pct($0) },reset:0.84,help:"How much of the canvas the product fills.")
            check("Fill each view separately",$s.independentScale,help:s.exportFraming=="fixed" ? "Applies to centred crops only.":"Every centred view fills the chosen amount, so product size changes between angles. Off keeps one scale set by the widest angle.").disabled(s.exportFraming=="fixed")
        }
    }
    var cutoutGroup:some View {
        panelSection("Cutout",$openCutout) {
            HStack(spacing:8) {
                Text("Engine").font(.system(size:13)).frame(width:70,alignment:.leading)
                Picker("Engine",selection:$s.maskBackend) { Text("Metal (GPU)").tag("mps");Text("CPU (ONNX)").tag("onnx") }.labelsHidden()
            }.help("Metal runs BiRefNet on the Apple Silicon GPU. CPU is the original rembg path; use it if Metal ever disagrees on a cutout. Masks are cached separately per engine.")
            HStack(spacing:8) {
                Text("Detection").font(.system(size:13)).frame(width:70,alignment:.leading)
                Picker("Detection",selection:$s.maskMode) { Text("Efficient").tag("efficient");Text("Detailed").tag("detailed") }.labelsHidden()
            }.help("Efficient uses a fast guide and one model pass. Detailed uses two passes.")
            check("Find small loose pieces",$s.smallParts,help:"For sets with studs, bones or tiles loose on the floor. Slower, and masks are rebuilt. Without it, pale pieces lying apart from the model can be missed.")
            check("Recover flat parts and tools",$s.recoverParts,help:"Extra close-up detection for missing ramps, grey plates and thin dark tools. Slower on first use; may keep more floor near dark pieces.")
            check("Recover white parts",$s.recoverWhite,help:"For photos where a white wing or plate merged into the booth. Check against Original. Applies to checked photos.")
            check("Recover dark parts near the floor",$s.recoverDark,help:"For black bumpers or plates cut away as shadow. Only long strips directly under the model are restored. Check against Original. Applies to checked photos.")
        }
    }
    /// Where the next export goes; in a folder's photo view, its existing export.
    var destinationPath:String? {
        if s.queueVisible { return s.queue.output_root }
        if let id=s.previewFolderID { return s.queue.jobs.first { $0.id==id }?.output }
        return s.outputParent.path
    }
    /// "Drive › Photos › Exports" for /Volumes/Drive/Photos/Exports/Batch-01; "~ › …" inside the home folder.
    func prettyParent(_ path:String)->String {
        var parts=URL(fileURLWithPath:path).deletingLastPathComponent().pathComponents.filter { $0 != "/" }
        let home=FileManager.default.homeDirectoryForCurrentUser.pathComponents.filter { $0 != "/" }
        if parts.first=="Volumes",parts.count>1 { parts.removeFirst() }
        else if parts.starts(with:home) { parts=["~"]+parts.dropFirst(home.count) }
        return parts.joined(separator:" › ")
    }
    var destinationCard:some View {
        let path=destinationPath ?? ""
        let available = !path.isEmpty && FileManager.default.fileExists(atPath:URL(fileURLWithPath:path).deletingLastPathComponent().path)
        let updating = !s.queueVisible && s.previewFolderID != nil
        let volume=path.hasPrefix("/Volumes/") ? URL(fileURLWithPath:path).pathComponents.dropFirst(2).first ?? "drive":"folder"
        return VStack(alignment:.leading,spacing:5) {
            Text(updating ? "Updates export":"Export to").font(.system(size:11)).foregroundStyle(muted)
            Button { if s.queueVisible { s.chooseQueueDestination() } else if !updating { s.chooseDestination() } } label: {
                HStack(spacing:9) {
                    Image(systemName:available ? "folder.fill":"externaldrive.badge.exclamationmark").font(.system(size:17))
                        .foregroundStyle(available ? accent:Color.yellow).frame(width:22)
                    VStack(alignment:.leading,spacing:1) {
                        Text(path.isEmpty ? "Choose a folder":URL(fileURLWithPath:path).lastPathComponent).font(.system(size:13,weight:.semibold)).lineLimit(1).truncationMode(.middle)
                        Text(available ? prettyParent(path):"\(volume) not connected").font(.system(size:11)).foregroundStyle(available ? muted:Color.yellow).lineLimit(1).truncationMode(.head)
                    }
                    Spacer(minLength:4)
                    if !updating { Text("Change").font(.system(size:11)).foregroundStyle(muted) }
                }.padding(.horizontal,10).padding(.vertical,8)
                    .background(RoundedRectangle(cornerRadius:8).fill(Color.white.opacity(0.05)))
                    .overlay(RoundedRectangle(cornerRadius:8).stroke(available ? Color.white.opacity(0.1):Color.yellow.opacity(0.5),lineWidth:1))
            }.buttonStyle(ToolButtonStyle(radius:8)).disabled(s.queueRunning || s.running)
                .help(path.isEmpty ? "Choose where exports are saved":path+(updating ? "":"\nClick to choose another folder."))
                .contextMenu {
                    Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath:path)]) }.disabled(!available)
                    if !updating { Button("Choose another folder…") { if s.queueVisible { s.chooseQueueDestination() } else { s.chooseDestination() } } }
                }
        }
    }
    var processTitle:String {
        if s.queueVisible && s.canResume { return "Resume · \(s.runFoldersLeft) \(s.runFoldersLeft==1 ? "folder":"folders") left" }
        if s.queueVisible {
            let n=s.exportableBatches.count
            if n==0 { return "Select folders to process" }
            let again=s.exportableBatches.allSatisfy { ["complete","error","stopped"].contains($0.status) }
            return "\(again ? "Re-export":"Process") \(n) \(n==1 ? "folder":"folders")"
        }
        let n=s.checked.count
        if n==0 { return "Select photos to process" }
        return "\(s.previewFolderID != nil ? "Update":"Process") \(n) \(n==1 ? "photo":"photos")"
    }
    var processDisabled:Bool { s.running || s.queueRunning || s.learningBusy || (s.queueVisible ? s.exportableBatches.isEmpty:s.checked.isEmpty) }
    var processArea:some View {
        VStack(spacing:10) {
            destinationCard
            Button(action:{ if s.queueVisible { s.canResume ? s.resumeQueue():s.startQueue() } else { s.run() } }) {
                Text(processTitle).font(.system(size:13,weight:.semibold)).frame(maxWidth:.infinity).padding(.vertical,10)
            }.buttonStyle(AccentButtonStyle(radius:8)).disabled(processDisabled)
                .help(s.queueVisible ? "Finished folders are exported again into their existing export folder. Keep the Mac plugged in with the lid open while processing.":"Writes JPEG and transparent PNG files. \(s.previewFolderID != nil ? "Updates the existing export for the checked photos.":"")")
        }.padding(16)
    }

    // MARK: footer
    var showFooter:Bool {
        s.running || (s.queueVisible && (s.queueRunning || s.canResume || (s.runSeen && ["complete","stopped"].contains(s.queue.status))))
    }
    var runSummary:String {
        let done=s.runPhotoDone,total=s.runPhotoTotal
        return "\(done.formatted())/\(total.formatted()) photos · \(Int((s.runProgress*100).rounded(.down)))%"
    }
    var elapsedLabel:some View {
        TimelineView(.periodic(from:.now,by:1)) { context in
            if let elapsed=s.queueElapsed(context.date) {
                Text(Studio.duration(elapsed)).font(.system(size:11,design:.monospaced)).foregroundStyle(muted).help("Elapsed time of this run")
            }
        }
    }
    /// The folder being processed, photo by photo: cells, this photo's timer and the last photo's time.
    var photoProgressLine:some View {
        let job=s.activeQueueJob
        let total=s.queue.photo_total ?? job?.sources.count ?? 0
        let number=s.queue.photo_number ?? 0
        let preparing=s.queue.photo_stage=="preparation"
        return HStack(spacing:10) {
            if total>0 {
                PhotoSegments(total:total,written:min(total,job?.completed ?? 0),current:number>0 ? number-1:nil,
                              currentProgress:s.queue.photo_progress ?? 0,preparing:preparing)
                    .frame(width:min(300,CGFloat(total)*8))
                    .help(preparing ? "Preparing cutouts (lighter) before writing photos (blue).":"Written photos are blue; the current photo fills as it renders.")
            }
            TimelineView(.periodic(from:.now,by:1)) { context in
                HStack(spacing:8) {
                    if number>0 { Text("\(preparing ? "Preparing":"Photo") \(min(number,max(1,total)))/\(total) · \(Studio.duration(s.queuePhotoElapsed(context.date)))").monospacedDigit() }
                    if !preparing,let last=s.queue.last_render_seconds,last>0 { Text("last \(Int(last.rounded())) s").monospacedDigit().help("Time the previous photo took to render") }
                }
            }.fixedSize()
            if let phase=s.queue.phase { Text(phase).lineLimit(1).truncationMode(.tail) }
        }.font(.system(size:11)).foregroundStyle(muted)
    }
    @ViewBuilder var queueFooter:some View {
        if s.queueRunning {
            VStack(alignment:.leading,spacing:3) {
                HStack(spacing:6) {
                    if let position=s.runFolderPosition { Text(position).foregroundStyle(muted) }
                    Text(s.activeQueueJob?.name ?? "Preparing queue").fontWeight(.medium).lineLimit(1).truncationMode(.middle)
                }.font(.system(size:12))
                photoProgressLine
            }
            Spacer(minLength:16)
            Text(runSummary).font(.system(size:12,design:.monospaced)).monospacedDigit()
            elapsedLabel
            if s.pauseRequested {
                HStack(spacing:6) { ProgressView().controlSize(.small);Text("Pausing…").font(.system(size:12)) }
                    .help("Pausing after the photo being written. Finished photos are kept.")
            } else {
                Button(action:s.pauseQueue) { Text("Pause").font(.system(size:12)).padding(.horizontal,10).padding(.vertical,4).background(RoundedRectangle(cornerRadius:6).fill(Color.white.opacity(0.08))) }.buttonStyle(ToolButtonStyle()).help("Stop after the current photo and keep everything finished; resume later.")
            }
            Button(role:.destructive) { confirmStop=true } label: { Text("Stop").font(.system(size:12)).padding(.horizontal,10).padding(.vertical,4).background(RoundedRectangle(cornerRadius:6).fill(Color.white.opacity(0.08))) }.buttonStyle(ToolButtonStyle())
                .confirmationDialog("Stop this export?",isPresented:$confirmStop) {
                    Button("Stop and discard export",role:.destructive,action:s.stopQueue)
                    Button("Keep processing",role:.cancel) {}
                } message:{ Text("The folder being processed stops now and its unfinished export moves to the Trash. Folders not started yet stay in the queue. Use Pause instead to keep finished photos and continue later.") }
        } else if s.canResume {
            Image(systemName:"pause.circle.fill").foregroundStyle(muted)
            VStack(alignment:.leading,spacing:3) {
                Text("Paused · \(s.runFoldersLeft) \(s.runFoldersLeft==1 ? "folder":"folders") left").font(.system(size:12,weight:.medium))
                Text(runSummary+(s.queue.phase.map { " · "+$0 } ?? "")).font(.system(size:11)).foregroundStyle(muted).lineLimit(1)
            }
            Spacer()
            elapsedLabel
            Button(action:s.resumeQueue) {
                Label("Resume",systemImage:"play.fill").font(.system(size:12,weight:.semibold)).padding(.horizontal,12).padding(.vertical,5)
            }.buttonStyle(AccentButtonStyle(radius:6)).disabled(s.running || s.learningBusy).help("Continue with the paused folder and the folders after it")
        } else {
            Image(systemName:s.queue.status=="complete" ? "checkmark.circle.fill":"stop.circle").foregroundStyle(s.queue.status=="complete" ? Color.green:muted)
            Text(s.queue.status=="complete" ? "Finished · \(s.runJobs.count) \(s.runJobs.count==1 ? "folder":"folders") · \(s.runPhotoTotal.formatted()) photos":(s.queue.phase ?? "Stopped"))
                .font(.system(size:12)).lineLimit(1)
            if s.queue.jobs.contains(where:{$0.errors>0 && $0.include != false}) {
                Text("· \(s.queue.jobs.filter{$0.include != false}.reduce(0){$0+$1.errors}) failed").font(.system(size:12)).foregroundStyle(.yellow)
            }
            Spacer()
            elapsedLabel
        }
    }
    var footer:some View {
        HStack(spacing:14) {
            if s.queueVisible {
                queueFooter
            } else if s.running {
                VStack(alignment:.leading,spacing:3) {
                    Text("Photo \(s.activePhoto) of \(s.batchTotal)").font(.system(size:12,weight:.medium))
                    Text(s.phase).font(.system(size:11)).foregroundStyle(muted).lineLimit(1)
                }
                Spacer()
                Text("\(s.batchCompleted)/\(s.batchTotal) photos · \(Int((Double(s.batchCompleted)/Double(max(1,s.batchTotal))*100).rounded(.down)))%").font(.system(size:12,design:.monospaced))
                Text(Studio.duration(s.batchElapsed)).font(.system(size:11,design:.monospaced)).foregroundStyle(muted)
                Button("Stop",action:s.cancel)
            }
        }.padding(.horizontal,16).frame(height:54).frame(maxWidth:.infinity).background(Color.black.opacity(0.18))
            .overlay(alignment:.top) {
                // Whole-run progress, counted from exported photos; only while processing.
                if s.queueVisible ? s.queueRunning:s.running {
                    GeometryReader { geo in
                        let fraction=s.queueVisible ? s.runProgress:Double(s.batchCompleted)/Double(max(1,s.batchTotal))
                        ZStack(alignment:.leading) {
                            Rectangle().fill(Color.white.opacity(0.08))
                            Rectangle().fill(accent).frame(width:geo.size.width*min(1,max(0,fraction)))
                        }
                    }.frame(height:3).animation(.easeOut(duration:0.3),value:s.queueVisible ? s.runProgress:Double(s.batchCompleted))
                }
            }
    }
}
@MainActor final class BrickStudioDelegate:NSObject,NSApplicationDelegate {
    static var cleanups:[String:()->Void]=[:]
    func applicationShouldTerminateAfterLastWindowClosed(_ sender:NSApplication)->Bool { true }
    func applicationWillTerminate(_ notification:Notification) {
        for cleanup in Array(Self.cleanups.values) { cleanup() }
    }
}
@main struct BrickStudioApp:App {
    @NSApplicationDelegateAdaptor(BrickStudioDelegate.self) var delegate
    // Tooltips appear after 0.3 s instead of the system's ~1 s.
    init() { UserDefaults.standard.register(defaults:["NSInitialToolTipDelay":300]) }
    var body:some Scene {
        WindowGroup("Brick Studio") { StudioView() }.defaultSize(width:1440,height:960).windowStyle(.hiddenTitleBar)
    }
}
