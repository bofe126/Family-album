import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:photo_view/photo_view.dart';
import 'face_recognition_service.dart'; // 添加这一行
import 'photo_state.dart'; // 添加这一行
import 'dart:io';
import 'package:file_picker/file_picker.dart';
import 'package:provider/provider.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  try {
    await FaceRecognitionService.initialize();
    print('FaceRecognitionService initialized successfully');
  } catch (e) {
    print('Failed to initialize FaceRecognitionService: $e');
  }
  
  runApp(
    ChangeNotifierProvider(
      create: (context) => PhotoState(),
      child: MyApp(),
    ),
  );
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '照片管理应用',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: PhotoViewer(),
    );
  }
}

class PhotoViewer extends StatelessWidget {
  final ImagePicker _picker = ImagePicker();

  void _pickImages(BuildContext context) async {
    final List<XFile>? selectedImages = await _picker.pickMultiImage();
    if (selectedImages != null && selectedImages.isNotEmpty) {
      Provider.of<PhotoState>(context, listen: false).addImages(selectedImages);
    }
  }

  void _pickImagesFromFilePicker(BuildContext context) async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      allowMultiple: true,
      type: FileType.image,
    );

    if (result != null) {
      List<XFile> selectedImages = result.paths.map((path) => XFile(_normalizePath(path!))).toList();
      Provider.of<PhotoState>(context, listen: false).addImages(selectedImages);
    }
  }

  void _onDragAccept(BuildContext context, List<DragTargetDetails<File>> details) {
    List<XFile> newImages = details
        .map((detail) => XFile(_normalizePath(detail.data.path)))
        .toList();
    Provider.of<PhotoState>(context, listen: false).addImages(newImages);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('照片浏览'),
        actions: [
          IconButton(
            icon: Icon(Icons.add_a_photo),
            onPressed: () => _pickImages(context),
          ),
          IconButton(
            icon: Icon(Icons.folder_open),
            onPressed: () => _pickImagesFromFilePicker(context),
          ),
        ],
      ),
      body: Consumer<PhotoState>(
        builder: (context, photoState, child) {
          return Row(
            children: [
              FaceSidebar(
                uniqueFaces: photoState.uniqueFaces,
                onFaceSelected: (face) => photoState.selectFace(face),
              ),
              Expanded(
                child: DragTarget<File>(
                  onAcceptWithDetails: (details) => _onDragAccept(context, [details]),
                  builder: (context, candidateData, rejectedData) {
                    return photoState.filteredImages.isEmpty
                        ? Center(child: Text('没有选择照片'))
                        : GridView.builder(
                            gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                              crossAxisCount: 3,
                            ),
                            itemCount: photoState.filteredImages.length,
                            itemBuilder: (context, index) {
                              return Stack(
                                children: [
                                  PhotoView(
                                    imageProvider: FileImage(File(photoState.filteredImages[index].path)),
                                  ),
                                  if (photoState.faceDetected(photoState.filteredImages[index].path))
                                    Positioned(
                                      right: 5,
                                      top: 5,
                                      child: Icon(Icons.face, color: Colors.green),
                                    ),
                                ],
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
    );
  }
}

class FaceSidebar extends StatelessWidget {
  final List<UniqueFace> uniqueFaces;
  final Function(UniqueFace) onFaceSelected;

  FaceSidebar({required this.uniqueFaces, required this.onFaceSelected});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 100,
      child: ListView.builder(
        itemCount: uniqueFaces.length,
        itemBuilder: (context, index) {
          return GestureDetector(
            onTap: () => onFaceSelected(uniqueFaces[index]),
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
}

String _normalizePath(String path) {
  return path.replaceAll('\\', '/');
}
