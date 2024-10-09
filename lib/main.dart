import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:photo_view/photo_view.dart';
import 'face_recognition_service.dart';
import 'photo_state.dart';
import 'dart:io';
import 'package:file_picker/file_picker.dart';
import 'package:provider/provider.dart';
import 'package:file_selector/file_selector.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:ffi/ffi.dart';
import 'dart:ffi';
import 'package:win32/win32.dart' as win32;
import 'logger.dart';

// 定义DLL函数类型（仅用于Windows）
typedef DetectFacesFunc = Pointer<Utf8> Function(Pointer<Utf8>);
typedef DetectFacesDart = Pointer<Utf8> Function(Pointer<Utf8>);

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await FaceRecognitionService.initialize();
  
  // 使用全局 logger 实例
  logger.i('应用程序启动');
  FlutterError.onError = (FlutterErrorDetails details) {
    logger.e('未捕获的Flutter错误: ${details.exception}');
  };
  
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (context) => PhotoState(),
      child: MaterialApp(
        title: '照片管理应用',
        theme: ThemeData(
          primarySwatch: Colors.blue,
        ),
        home: PhotoViewer(),
      ),
    );
  }
}

class PhotoViewer extends StatefulWidget {
  const PhotoViewer({Key? key}) : super(key: key);

  @override
  State<PhotoViewer> createState() => _PhotoViewerState();
}

class _PhotoViewerState extends State<PhotoViewer> {
  final ImagePicker _picker = ImagePicker();

  void _pickImages() async {
    final List<XFile> selectedImages = await _picker.pickMultiImage();
    if (!mounted) return; // 检查组件是否仍然挂载
    if (selectedImages.isNotEmpty) {
      Provider.of<PhotoState>(context, listen: false).addImages(selectedImages);
    }
  }

  void _pickImagesFromFilePicker(BuildContext context) async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      allowMultiple: true,
      type: FileType.image,
    );

    if (result != null) {
      if (result.paths.isNotEmpty) {
        final selectedImages =
            result.paths.map((path) => XFile(_normalizePath(path!))).toList();
        if (context.mounted) {
          Provider.of<PhotoState>(context, listen: false)
              .addImages(selectedImages);
        }
      }
    }
  }

  void _onDragAccept(
      BuildContext context, List<DragTargetDetails<File>> details) {
    List<XFile> newImages = details
        .map((detail) => XFile(_normalizePath(detail.data.path)))
        .toList();
    Provider.of<PhotoState>(context, listen: false).addImages(newImages);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('照片浏览'),
        actions: [
          IconButton(
            icon: const Icon(Icons.add_a_photo),
            onPressed: () => _pickImages(),
          ),
          IconButton(
            icon: const Icon(Icons.folder_open),
            onPressed: () => _pickImagesFromFilePicker(context),
          ),
        ],
      ),
      body: Column(
        children: [
          ThresholdSlider(),
          SearchBar(),
          Expanded(
            child: Consumer<PhotoState>(
              builder: (context, photoState, child) {
                debugPrint("Total images: ${photoState.imageFileList.length}");
                debugPrint(
                    "Filtered images: ${photoState.filteredImages.length}");
                debugPrint(
                    "Selected face: ${photoState.selectedFace?.label ?? 'None'}");
                debugPrint("Search query: ${photoState.searchQuery}");
                return Row(
                  children: [
                    FaceSidebar(
                      uniqueFaces: photoState.uniqueFaces,
                      onFaceSelected: (face) => photoState.selectFace(face),
                      onFaceLabeled: (face, label) =>
                          photoState.labelFace(face, label),
                    ),
                    Expanded(
                      child: DragTarget<File>(
                        onAcceptWithDetails: (details) =>
                            _onDragAccept(context, [details]),
                        builder: (context, candidateData, rejectedData) {
                          return photoState.filteredImages.isEmpty
                              ? const Center(child: Text('没有选择照片'))
                              : GridView.builder(
                                  gridDelegate:
                                      const SliverGridDelegateWithFixedCrossAxisCount(
                                    crossAxisCount: 3,
                                  ),
                                  itemCount: photoState.filteredImages.length,
                                  itemBuilder: (context, index) {
                                    return GestureDetector(
                                      onTap: () => _showFullScreenImage(context,
                                          photoState.filteredImages[index]),
                                      child: Stack(
                                        children: [
                                          Image.file(
                                            File(photoState
                                                .filteredImages[index].path),
                                            fit: BoxFit.cover,
                                          ),
                                          if (photoState.faceDetected(photoState
                                              .filteredImages[index].path))
                                            Positioned(
                                              right: 5,
                                              top: 5,
                                              child: Icon(Icons.face,
                                                  color: Colors.green),
                                            ),
                                        ],
                                      ),
                                    );
                                  },
                                );
                        },
                      ),
                    ),
                  ],
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  void _showFullScreenImage(BuildContext context, XFile image) {
    Navigator.of(context).push(MaterialPageRoute(
      builder: (context) => Scaffold(
        body: Center(
          child: PhotoView(
            imageProvider: FileImage(File(image.path)),
          ),
        ),
      ),
    ));
  }
}

class ThresholdSlider extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Consumer<PhotoState>(
      builder: (context, photoState, child) {
        return Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16.0),
          child: Row(
            children: [
              Text('相似度阈值:'),
              Expanded(
                child: Slider(
                  value: photoState.similarityThreshold,
                  min: 0.5,
                  max: 1.0,
                  divisions: 50,
                  label: photoState.similarityThreshold.toStringAsFixed(2),
                  onChanged: (double value) {
                    photoState.setSimilarityThreshold(value);
                  },
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class SearchBar extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: TextField(
        decoration: InputDecoration(
          hintText: '搜索人名...',
          prefixIcon: Icon(Icons.search),
          border: OutlineInputBorder(),
        ),
        onChanged: (value) {
          Provider.of<PhotoState>(context, listen: false).setSearchQuery(value);
        },
      ),
    );
  }
}

class FaceSidebar extends StatelessWidget {
  final List<UniqueFace> uniqueFaces;
  final Function(UniqueFace) onFaceSelected;
  final Function(UniqueFace, String) onFaceLabeled;

  FaceSidebar(
      {required this.uniqueFaces,
      required this.onFaceSelected,
      required this.onFaceLabeled});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 100,
      child: ListView.builder(
        itemCount: uniqueFaces.length,
        itemBuilder: (context, index) {
          return GestureDetector(
            onTap: () => onFaceSelected(uniqueFaces[index]),
            onLongPress: () => _showLabelDialog(context, uniqueFaces[index]),
            child: Stack(
              alignment: Alignment.bottomRight,
              children: [
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Image.memory(uniqueFaces[index].faceData.faceImage),
                ),
                Container(
                  padding: EdgeInsets.all(2),
                  color: Colors.black54,
                  child: Text(
                    uniqueFaces[index].label ??
                        '${uniqueFaces[index].images.length}',
                    style: TextStyle(color: Colors.white, fontSize: 10),
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }

  void _showLabelDialog(BuildContext context, UniqueFace face) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        String newLabel = face.label ?? '';
        return AlertDialog(
          title: Text('标记人脸'),
          content: TextField(
            onChanged: (value) {
              newLabel = value;
            },
            decoration: InputDecoration(hintText: "输入名字"),
          ),
          actions: <Widget>[
            TextButton(
              child: Text('取消'),
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
            TextButton(
              child: Text('确定'),
              onPressed: () {
                onFaceLabeled(face, newLabel);
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }
}

String _normalizePath(String path) {
  return path.replaceAll('\\', '/');
}
