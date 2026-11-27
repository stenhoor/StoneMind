// @stonemind/files: the file kit. The FileStore port on OPFS, path rules, and browser helpers.
export { PathError, isWithin, join, nameOf, normalize, parentOf, segments } from './paths.js';
export { type ChangeDetail, NotFoundError, OpfsFileStore, type OpfsFileStoreOptions, type WriteMethod } from './opfs.js';
export { type ImportProgress, type StorageInfo, downloadBlob, filesFromDrop, importFiles, pickFiles, requestPersistence, storageInfo } from './browser.js';
export type { FileData, FileEntry, FileStore, FileWriteOptions } from '@stonemind/storage';
